import pytest

from env import TicketTriageEnv


class TestTicketTriageEnv:
    def test_invalid_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid task"):
            TicketTriageEnv(task="invalid")

    def test_easy_initial_state(self) -> None:
        env = TicketTriageEnv(task="easy")
        state = env.reset()
        assert state["status"] == "in_progress"
        assert state["pending_ticket"] == "My screen is cracked."

    def test_easy_correct_routing(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        _, reward, done, _ = env.step("repair")
        assert reward == 1.0
        assert done
        assert env.grader() == 1.0

    def test_easy_wrong_routing(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        _, reward, done, _ = env.step("billing")
        assert reward == 0.0
        assert done
        assert env.grader() == 0.0

    def test_medium_all_correct(self) -> None:
        env = TicketTriageEnv(task="medium")
        env.reset()
        for action in ["repair", "billing", "sales"]:
            env.step(action)
        assert env.grader() == 1.0

    def test_medium_partial(self) -> None:
        env = TicketTriageEnv(task="medium")
        env.reset()
        env.step("repair")
        env.step("tech")
        env.step("sales")
        assert env.grader() == 0.67

    def test_hard_all_correct(self) -> None:
        env = TicketTriageEnv(task="hard")
        env.reset()
        for action in ["repair", "billing", "sales", "tech", "billing"]:
            env.step(action)
        assert env.grader() == 1.0

    def test_hard_partial(self) -> None:
        env = TicketTriageEnv(task="hard")
        env.reset()
        for action in ["tech", "tech", "tech", "tech", "tech"]:
            env.step(action)
        assert env.grader() == 0.2

    def test_step_after_done_returns_zero_reward(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        env.step("repair")
        _, reward, done, info = env.step("repair")
        assert done
        assert reward == 0.0
        assert "error" in info

    def test_state_complete_after_all_routed(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        env.step("repair")
        state = env.state()
        assert state["status"] == "complete"
        assert state["pending_ticket"] is None

    def test_action_case_insensitive(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        _, reward, _, _ = env.step("REPAIR")
        assert reward == 1.0

    def test_action_with_whitespace(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        _, reward, _, _ = env.step("  repair  ")
        assert reward == 1.0

    def test_total_steps_tracked(self) -> None:
        env = TicketTriageEnv(task="medium")
        env.reset()
        env.step("repair")
        env.step("billing")
        assert env.total_steps == 2

    def test_reset_clears_state(self) -> None:
        env = TicketTriageEnv(task="medium")
        env.reset()
        env.step("repair")
        env.step("billing")
        env.reset()
        assert env.current_ticket_idx == 0
        assert env.correct_routing == 0
        assert env.total_steps == 0
        assert not env.done

    def test_correct_routing_count(self) -> None:
        env = TicketTriageEnv(task="medium")
        env.reset()
        env.step("repair")
        env.step("wrong")
        env.step("sales")
        assert env.correct_routing == 2

    def test_step_returns_tuple_of_four(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        result = env.step("repair")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_reward_is_float(self) -> None:
        env = TicketTriageEnv(task="medium")
        env.reset()
        _, reward, _, _ = env.step("repair")
        assert isinstance(reward, float)

    def test_grader_zero_tickets(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.tickets = []
        assert env.grader() == 0.0

    def test_grader_before_any_steps(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        assert env.grader() == 0.0

    def test_valid_tasks_are_tuple(self) -> None:
        assert isinstance(TicketTriageEnv.VALID_TASKS, tuple)

    def test_valid_departments_are_tuple(self) -> None:
        assert isinstance(TicketTriageEnv.VALID_DEPARTMENTS, tuple)

    def test_departments_cover_all_tickets(self) -> None:
        env = TicketTriageEnv(task="hard")
        env.reset()
        ticket_depts = {str(t["dept"]) for t in env.tickets}
        assert ticket_depts.issubset(set(TicketTriageEnv.VALID_DEPARTMENTS))

    def test_reward_fractional_easy(self) -> None:
        env = TicketTriageEnv(task="easy")
        env.reset()
        _, reward, _, _ = env.step("repair")
        assert reward == 1.0

    def test_reward_fractional_medium(self) -> None:
        env = TicketTriageEnv(task="medium")
        env.reset()
        _, reward, _, _ = env.step("repair")
        assert reward == 0.33

    def test_reward_fractional_hard(self) -> None:
        env = TicketTriageEnv(task="hard")
        env.reset()
        _, reward, _, _ = env.step("repair")
        assert reward == pytest.approx(1.0 / 5.0)
