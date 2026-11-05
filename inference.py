import argparse
import logging
import os
import sys

from openai import OpenAI

from env import TicketTriageEnv

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer support ticket triage inference runner")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["easy", "medium", "hard"],
        choices=TicketTriageEnv.VALID_TASKS,
        help=f"Tasks to run (default: all; choices: {', '.join(TicketTriageEnv.VALID_TASKS)})",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        help="Model name (default: gpt-3.5-turbo, or MODEL_NAME env var)",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
        help="API base URL (default: https://api.openai.com/v1, or API_BASE_URL env var)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", os.getenv("HF_TOKEN", "")),
        help="API key (default: OPENAI_API_KEY or HF_TOKEN env var)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def run_inference(task_name: str, client: OpenAI, model_name: str) -> float:
    env = TicketTriageEnv(task=task_name)
    env.reset()

    logger.info("[START] task=%s env=customer-support-triage model=%s", task_name, model_name)

    step_count = 0
    rewards_history: list[float] = []

    while not env.done:
        step_count += 1
        ticket_text: str = str(env.state().get("pending_ticket", ""))

        action = "unknown"
        error_str = "null"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You route tickets. Reply ONLY with one of: repair, billing, sales, tech.",
                    },
                    {"role": "user", "content": ticket_text},
                ],
                timeout=30,
            )
            content = response.choices[0].message.content
            action = content.strip().lower() if content else "unknown"
        except Exception as e:
            error_str = str(e).replace("\n", " ")
            logger.error("API call failed at step %d: %s", step_count, error_str)

        _, reward, done, _ = env.step(action)
        rewards_history.append(reward)

        logger.info(
            "[STEP] step=%d action=%s reward=%.2f done=%s error=%s success=%s steps=%d score=%.2f rewards=%s",
            step_count,
            action,
            reward,
            "true" if done else "false",
            error_str,
            "true" if env.grader() == 1.0 else "false",
            step_count,
            env.grader(),
            ",".join(str(r) for r in rewards_history),
        )

    final_score = env.grader()
    logger.info(
        "[END] task=%s score=%.2f steps=%d correct=%d total=%d",
        task_name,
        final_score,
        step_count,
        env.correct_routing,
        len(env.tickets),
    )
    return final_score


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    if not args.api_key:
        logger.error(
            "No API key found. Set OPENAI_API_KEY (or HF_TOKEN) environment variable "
            "or pass --api-key."
        )
        sys.exit(1)

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    all_scores: dict[str, float] = {}
    for task in args.tasks:
        logger.info("=" * 60)
        logger.info("Running task: %s", task)
        logger.info("=" * 60)
        score = run_inference(task, client, args.model)
        all_scores[task] = score

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for task, score in all_scores.items():
        status = "PASS" if score == 1.0 else "FAIL"
        logger.info("  %s: score=%.2f [%s]", task, score, status)


if __name__ == "__main__":
    main()
