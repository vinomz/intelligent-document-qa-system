import asyncio
import aiohttp
import time
import numpy as np
from config import settings

URL = settings.AI_RAG_API_URL + "query"
CONCURRENCY = 20
DURATION_SECONDS = 120

async def worker(session, latencies, errors):
    end_time = time.time() + DURATION_SECONDS
    while time.time() < end_time:
        start = time.time()
        try:
            async with session.post(URL, json={"question": "test"}) as resp:
                await resp.text()
                if resp.status != 200:
                    errors.append(resp.status)
        except Exception as e:
            errors.append(str(e))
        finally:
            latencies.append(time.time() - start)

async def run_test():
    latencies = []
    errors = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(worker(session, latencies, errors))
            for _ in range(CONCURRENCY)
        ]
        await asyncio.gather(*tasks)

    # Convert to numpy array for percentile calculations
    lat_arr = np.array(latencies)

    p50 = np.percentile(lat_arr, 50)
    p90 = np.percentile(lat_arr, 90)
    p95 = np.percentile(lat_arr, 95)

    return {
        "count": len(lat_arr),
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "errors": errors
    }
