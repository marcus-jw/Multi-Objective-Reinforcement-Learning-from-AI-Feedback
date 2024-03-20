import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field
from vertexai.generative_models import GenerativeModel
import vertexai
from vertexai import generative_models
async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    api_endpoint: str,
    project_id: str,
    region: str,
    max_requests_per_minute: float,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    with open(requests_filepath) as file:
        # read all requests from the file
        requests = [json.loads(line) for line in file]
        logging.debug(f"File opened. Total requests: {len(requests)}")

    async with aiohttp.ClientSession() as session:
        while requests:
            # create a list to store the tasks
            tasks = []

            # create tasks for available requests
            while requests and len(tasks) < max_requests_per_minute:
                request_json = requests.pop(0)
                next_request = APIRequest(
                    task_id=next(task_id_generator),
                    request_json=request_json,
                    attempts_left=max_attempts,
                    metadata=request_json.pop("metadata", None),
                )
                status_tracker.num_tasks_started += 1
                status_tracker.num_tasks_in_progress += 1
                logging.debug(
                    f"Reading request {next_request.task_id}: {next_request}"
                )

                # create a task for the API request
                task = asyncio.create_task(
                    next_request.call_api(
                        session=session,
                        api_endpoint=api_endpoint,
                        project_id=project_id,
                        region=region,
                        retry_queue=queue_of_requests_to_retry,
                        save_filepath=save_filepath,
                        status_tracker=status_tracker,
                    )
                )
                tasks.append(task)

            # wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks)

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity
                + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            
            last_update_time = current_time

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
    self,
    session: aiohttp.ClientSession,
    api_endpoint: str,
    project_id: str,
    region: str,
    retry_queue: asyncio.Queue,
    save_filepath: str,
    status_tracker: StatusTracker):
        """Calls the Vertex AI Gemini API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            # Initialize Vertex AI SDK
            vertexai.init(project=project_id, location=region)
            model = GenerativeModel("gemini-1.0-pro")
            safety_config = {generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                }
            # Make the API request
            asyncio.sleep(0.001)
            response = await asyncio.to_thread(
                        model.generate_content,
                        **self.request_json,
                        safety_settings=safety_config
                        )
            

            # Extract the relevant information from the Candidate object
            result = []
            for candidate in response.candidates:
                result.append({
                    "content": {
                        "parts": [{"text": part.text} for part in candidate.content.parts]
                    },
                    "finishReason": candidate.finish_reason,
                    "safetyRatings": [
                        {
                            "category": rating.category,
                            "probability": rating.probability,
                            "blocked": rating.blocked
                        }
                        for rating in candidate.safety_ratings
                    ]
                })

        except Exception as e:
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(str(error))
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, self.result, self.metadata]
                    if self.metadata
                    else [self.request_json, self.result]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, result, self.metadata]
                if self.metadata
                else [self.request_json, result]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions

def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


# run script

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--api_endpoint", default="us-central1-aiplatform.googleapis.com")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--max_requests_per_minute", type=int, default=300)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--logging_level", default=logging.INFO)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            api_endpoint=args.api_endpoint,
            project_id=args.project_id,
            region=args.region,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
        )
    )