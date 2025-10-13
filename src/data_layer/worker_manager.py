import logging
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class WorkerInfo:
    """Information about a registered worker"""
    def __init__(self, name: str, worker_instance: Any, start_method: str, stop_method: str):
        self.name = name
        self.worker_instance = worker_instance
        self.start_method = start_method
        self.stop_method = stop_method
        self.started = False
        self.start_time = None
        self.stop_time = None
        self.error_count = 0

class WorkerManager:
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = WorkerManager()
        return cls._instance
    
    def __init__(self):
        self._workers: Dict[str, WorkerInfo] = {}
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._running = False
    
    def register_worker(self, 
                      name: str, 
                      worker_instance: Any, 
                      start_method: str = "start", 
                      stop_method: str = "stop") -> bool:
        with self._lock:
            if name in self._workers:
                logger.warning(f"Worker {name} already registered")
                return False
            
            # Validate that worker has the required methods
            if not hasattr(worker_instance, start_method):
                logger.error(f"Worker {name} does not have start method {start_method}")
                return False
                
            if not hasattr(worker_instance, stop_method):
                logger.error(f"Worker {name} does not have stop method {stop_method}")
                return False
            
            # Create worker info
            worker_info = WorkerInfo(
                name=name,
                worker_instance=worker_instance,
                start_method=start_method,
                stop_method=stop_method
            )
            
            # Register worker
            self._workers[name] = worker_info
            logger.info(f"Worker {name} registered successfully")
            
            return True
    
    def unregister_worker(self, name: str) -> bool:
        with self._lock:
            if name not in self._workers:
                logger.warning(f"Worker {name} not registered")
                return False
            
            # Stop worker if running
            if self._workers[name].started:
                self.stop_worker(name)
            
            # Remove worker
            del self._workers[name]
            logger.info(f"Worker {name} unregistered successfully")
            
            return True
    
    def start_worker(self, name: str) -> bool:
        with self._lock:
            if name not in self._workers:
                logger.warning(f"Worker {name} not registered")
                return False
            
            worker_info = self._workers[name]
            if worker_info.started:
                logger.warning(f"Worker {name} already started")
                return True
            
            try:
                start_method = getattr(worker_info.worker_instance, worker_info.start_method)
                result = start_method()
                if result is None or result:  # If result is None or True, consider success
                    worker_info.started = True
                    worker_info.start_time = datetime.now()
                    worker_info.stop_time = None
                    logger.info(f"Worker {name} started successfully")
                    return True
                else:
                    logger.error(f"Worker {name} failed to start")
                    worker_info.error_count += 1
                    return False
                    
            except Exception as e:
                logger.error(f"Error starting worker {name}: {e}")
                worker_info.error_count += 1
                return False
    
    def stop_worker(self, name: str) -> bool:
        with self._lock:
            if name not in self._workers:
                logger.warning(f"Worker {name} not registered")
                return False
            
            worker_info = self._workers[name]
            if not worker_info.started:
                logger.warning(f"Worker {name} not started")
                return True
            
            try:
                stop_method = getattr(worker_info.worker_instance, worker_info.stop_method)
                stop_method()
                worker_info.started = False
                worker_info.stop_time = datetime.now()
                logger.info(f"Worker {name} stopped successfully")
                
                return True
                    
            except Exception as e:
                logger.error(f"Error stopping worker {name}: {e}")
                worker_info.error_count += 1
                return False
    
    def start_all_workers(self) -> Dict[str, bool]:
        results = {}
        
        for name in list(self._workers.keys()):
            results[name] = self.start_worker(name)
        
        # Log summary
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Started {success_count}/{len(results)} workers")
        
        return results
    
    def stop_all_workers(self) -> Dict[str, bool]:
        results = {}
        
        for name in list(self._workers.keys()):
            results[name] = self.stop_worker(name)

        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Stopped {success_count}/{len(results)} workers")
        
        return results
    
    def get_worker_status(self, name: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if name not in self._workers:
                return None
            
            worker_info = self._workers[name]
            additional_status = {}
            if hasattr(worker_info.worker_instance, "get_status"):
                try:
                    additional_status = worker_info.worker_instance.get_status()
                except Exception as e:
                    logger.error(f"Error getting additional status for worker {name}: {e}")
            
            status = {
                "name": name,
                "running": worker_info.started,
                "start_time": worker_info.start_time.isoformat() if worker_info.start_time else None,
                "stop_time": worker_info.stop_time.isoformat() if worker_info.stop_time else None,
                "uptime_seconds": (datetime.now() - worker_info.start_time).total_seconds() if worker_info.started and worker_info.start_time else 0,
                "error_count": worker_info.error_count
            }
            
            status.update(additional_status)
            
            return status
    
    def get_all_worker_status(self) -> Dict[str, Dict[str, Any]]:
        status = {}
        
        for name in list(self._workers.keys()):
            status[name] = self.get_worker_status(name)
        
        return status
    
    def start_monitoring(self, interval: float = 60.0):
        if self._running:
            logger.warning("Monitoring thread already running")
            return
            
        self._running = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Worker monitoring thread started")
    
    def stop_monitoring(self):
        if not self._running:
            return
            
        self._running = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
            if self._monitoring_thread.is_alive():
                logger.warning("Monitoring thread did not terminate gracefully")
        
        logger.info("Worker monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        logger.info(f"Worker monitoring started with {interval}s interval")
        
        while self._running:
            try:
                with self._lock:
                    for name, worker_info in self._workers.items():
                        if worker_info.started:
                            if not hasattr(worker_info.worker_instance, worker_info.start_method) or \
                               not hasattr(worker_info.worker_instance, worker_info.stop_method):
                                logger.error(f"Worker {name} missing required methods, marking as stopped")
                                worker_info.started = False
                                worker_info.stop_time = datetime.now()
                                worker_info.error_count += 1
                                continue
                        
                            if hasattr(worker_info.worker_instance, "is_alive"):
                                try:
                                    if not worker_info.worker_instance.is_alive():
                                        logger.error(f"Worker {name} is not alive, attempting to restart")
                                        self.stop_worker(name)
                                        self.start_worker(name)
                                except Exception as e:
                                    logger.error(f"Error checking worker {name} alive status: {e}")
            
            except Exception as e:
                logger.error(f"Error in worker monitoring: {e}")
            
            # Sleep for the specified interval
            time.sleep(interval)
        
        logger.info("Worker monitoring thread stopped")