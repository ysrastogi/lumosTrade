class SlippagePredictor:
    """Pre-trade cost analysis"""
    
    def __init__(self):
        self.historical_slippage: Dict[str, List[float]] = {}
    
    def predict_slippage(self, symbol: str, quantity: float, order_type: OrderType) -> float:
        """Predict expected slippage in basis points"""
        base_slippage = 0.5  # 0.5 bps base
        
        # Adjust for order size (larger orders = more slippage)
        size_factor = min(quantity / 1000, 5.0)
        
        # Adjust for order type
        type_factor = {
            OrderType.MARKET: 2.0,
            OrderType.LIMIT: 1.0,
            OrderType.STOP: 1.5,
            OrderType.ICEBERG: 0.8
        }[order_type]
        
        # Historical adjustment
        hist_avg = 0
        if symbol in self.historical_slippage and self.historical_slippage[symbol]:
            hist_avg = sum(self.historical_slippage[symbol][-20:]) / min(20, len(self.historical_slippage[symbol]))
        
        predicted = base_slippage * size_factor * type_factor + hist_avg * 0.3
        return round(predicted, 2)
    
    def record_actual_slippage(self, symbol: str, slippage_bps: float):
        """Record actual slippage for learning"""
        if symbol not in self.historical_slippage:
            self.historical_slippage[symbol] = []
        self.historical_slippage[symbol].append(slippage_bps)