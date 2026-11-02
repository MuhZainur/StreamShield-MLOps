"""High-Throughput Streaming Event Ingestion & Drift Injection Simulator."""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import random
from typing import AsyncGenerator, Dict, List


@dataclass
class TransactionEvent:
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    timestamp: str
    device_id: str
    ip_address: str
    is_fraud_ground_truth: int


class StreamingEventBroker:
    """In-memory distributed-compatible asynchronous ring buffer queue."""

    def __init__(self, maxsize: int = 10000):
        self._queue: asyncio.Queue[TransactionEvent] = asyncio.Queue(maxsize=maxsize)

    async def publish(self, event: TransactionEvent) -> None:
        await self._queue.put(event)

    async def consume(self) -> AsyncGenerator[TransactionEvent, None]:
        while True:
            event = await self._queue.get()
            yield event
            self._queue.task_done()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()


def generate_synthetic_event(
    event_idx: int,
    drift_mode: bool = False,
    user_pool_size: int = 500,
) -> TransactionEvent:
    """Generate single transaction event with optional concept/data drift."""
    user_id = f"usr_{random.randint(1, user_pool_size):05d}"
    merchant_id = f"merch_{random.randint(1, 50):03d}"
    now_iso = datetime.now(timezone.utc).isoformat()

    if drift_mode:
        # Botnet card-testing attack: micro amounts, rapid fire, high fraud rate
        amount = round(random.uniform(0.50, 4.99), 2)
        is_fraud = 1 if random.random() < 0.75 else 0
        ip_addr = f"10.99.{random.randint(1, 255)}.{random.randint(1, 255)}"
    else:
        # Normal customer behavior: gamma/lognormal amounts, low fraud rate (~2%)
        amount = round(random.gammavariate(2.0, 30.0), 2)
        is_fraud = 1 if random.random() < 0.02 else 0
        ip_addr = f"192.168.1.{random.randint(1, 100)}"

    return TransactionEvent(
        transaction_id=f"tx_{event_idx:08d}",
        user_id=user_id,
        merchant_id=merchant_id,
        amount=amount,
        timestamp=now_iso,
        device_id=f"dev_{random.randint(100, 999)}",
        ip_address=ip_addr,
        is_fraud_ground_truth=is_fraud,
    )


async def simulate_stream(
    broker: StreamingEventBroker,
    n_events: int = 1000,
    drift_ratio: float = 0.0,
    delay_ms: float = 0.0,
) -> None:
    """Stream transactions into the broker."""
    for i in range(n_events):
        is_drift = random.random() < drift_ratio
        event = generate_synthetic_event(i, drift_mode=is_drift)
        await broker.publish(event)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
