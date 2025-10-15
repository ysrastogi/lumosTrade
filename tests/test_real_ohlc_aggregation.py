import logging
import time
import logging
import time
from datetime import datetime

from src.data_layer.market_stream.stream import MarketStream
from src.data_layer.aggregator.market_aggregator import MarketDataAggregator

# === CONFIGURE LOGGING ===
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("REAL_OHLC_TEST")

# Optional: limit asyncio debug noise if present
logging.getLogger("asyncio").setLevel(logging.WARNING)


def run_real_ohlc_test():
    logger.info("=== Starting Real OHLC Aggregation Test ===")

    # Create a real MarketStream instance (with your actual config)
    market_stream = MarketStream()

    # Initialize aggregator
    aggregator = MarketDataAggregator(market_stream=market_stream)

    # Patch aggregator methods for deeper logging
    original_process_ohlc = aggregator._process_ohlc
    original_subscribe = market_stream.subscribe_ohlc

    # --- Step 1: Inject deep tracing in subscribe_ohlc ---
    def traced_subscribe_ohlc(symbol, interval):
        logger.info(f"📡 Subscribing to OHLC stream for {symbol} [{interval}] ...")
        try:
            result = original_subscribe(symbol, interval)
            logger.info(f"✅ Subscription successful for {symbol} [{interval}]")
            return result
        except Exception as e:
            logger.exception(f"❌ Subscription failed for {symbol} [{interval}]")
            raise e

    market_stream.subscribe_ohlc = traced_subscribe_ohlc

    # --- Step 2: Trace OHLC data processing flow ---
    def traced_process_ohlc(data):
        logger.debug(f"📥 Raw OHLC data received: {data}")
        try:
            before_count = 0
            symbol = data.get("ohlc", {}).get("symbol")
            if symbol:
                norm_symbol = aggregator._normalize_symbol(symbol).display
                before_count = len(aggregator._historical_cache.get(norm_symbol, {}).get("1m", []))

            original_process_ohlc(data)

            after_count = len(
                aggregator._historical_cache.get(norm_symbol, {}).get("1m", [])
            )
            if after_count > before_count:
                logger.info(
                    f"✅ OHLC successfully aggregated for {symbol} (Cache size: {after_count})"
                )
            else:
                logger.warning(
                    f"⚠️ OHLC received but not stored for {symbol}. Cache count unchanged ({after_count})"
                )
        except Exception as e:
            logger.exception(f"❌ Error inside _process_ohlc: {e}")

    aggregator._process_ohlc = traced_process_ohlc

    # --- Step 3: Start aggregator ---
    started = aggregator.start()
    if not started:
        logger.error("❌ Failed to start aggregator. Check MarketStream connectivity.")
        return

    logger.info("🚀 Aggregator started. Waiting for OHLC data...")

    # --- Step 4: Monitor live data aggregation ---
    watch_symbols = ["R_75", "R_50", "R_10", "R_100"]  # Change as per config
    check_interval = 10  # seconds

    try:
        while True:
            time.sleep(check_interval)
            logger.info("⏱ Checking cache & metrics...")
            for symbol in watch_symbols:
                norm_symbol = aggregator._normalize_symbol(symbol).display
                metrics = aggregator.get_symbol_metrics(symbol)
                ohlc_data = aggregator.get_historical_ohlc(symbol, "1m", limit=5)

                if metrics:
                    logger.info(
                        f"[{norm_symbol}] Last Price: {metrics.last_price:.4f}, "
                        f"Δ15m: {metrics.price_change_15m:.3f}%, "
                        f"Volatility: {metrics.volatility:.3f}"
                    )
                else:
                    logger.warning(f"⚠️ No metrics found for {norm_symbol}")

                logger.info(f"📊 OHLC stored: {len(ohlc_data)} entries")

                if ohlc_data:
                    last_ohlc = ohlc_data[-1]
                    logger.debug(
                        f"Last OHLC for {norm_symbol}: close={last_ohlc['close']}, time={last_ohlc['timestamp']}"
                    )
                else:
                    logger.debug(f"No OHLC data yet for {norm_symbol}")

    except KeyboardInterrupt:
        logger.info("🛑 Stopping test manually (Ctrl+C)...")
        aggregator.stop()


if __name__ == "__main__":
    run_real_ohlc_test()
