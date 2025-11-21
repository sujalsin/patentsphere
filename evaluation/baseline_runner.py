import json
from datetime import datetime

import asyncio
from app.orchestrator import Orchestrator


def run_baseline(queries):
    orchestrator = Orchestrator()
    results = []

    async def run_query(q):
        res = await orchestrator.run_all(q)
        return {"query": q, "results": {k: v.data for k, v in res.items()}}

    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(run_query(query)) for query in queries]
    loop.run_until_complete(asyncio.gather(*tasks))
    for t in tasks:
        results.append(t.result())

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "queries": results,
    }

    with open("evaluation/baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)

