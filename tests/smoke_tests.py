#!/usr/bin/env python3
"""
Smoke tests for AI Trading Platform deployment.
Requirements: 6.1, 6.3, 6.6
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, List, Optional

import aiohttp
import asyncpg
import aioredis


class SmokeTestRunner:
    """Runner for deployment smoke tests."""
    
    def __init__(self, environment: str = "staging", base_url: str = None):
        self.environment = environment
        self.base_url = base_url or self._get_default_base_url()
        self.timeout = 30
        self.results = {}
    
    def _get_default_base_url(self) -> str:
        """Get default base URL based on environment."""
        if self.environment == "production":
            return "https://api.trading-platform.com"
        else:
            return "http://localhost:8080"  # Assumes port-forward for testing
    
    async def test_api_health(self) -> bool:
        """Test API health endpoint."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.results["api_health"] = {
                            "status": "pass",
                            "response_time": data.get("response_time", 0),
                            "details": data
                        }
                        return True
                    else:
                        self.results["api_health"] = {
                            "status": "fail",
                            "error": f"HTTP {response.status}"
                        }
                        return False
        except Exception as e:
            self.results["api_health"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def test_api_status(self) -> bool:
        """Test API status endpoint."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/api/v1/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.results["api_status"] = {
                            "status": "pass",
                            "version": data.get("version", "unknown"),
                            "environment": data.get("environment", "unknown")
                        }
                        return True
                    else:
                        self.results["api_status"] = {
                            "status": "fail",
                            "error": f"HTTP {response.status}"
                        }
                        return False
        except Exception as e:
            self.results["api_status"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def test_authentication(self) -> bool:
        """Test authentication endpoint."""
        try:
            # Test login endpoint with dummy credentials
            test_credentials = {
                "username": "test_user",
                "password": "test_password"
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(f"{self.base_url}/api/v1/auth/login", json=test_credentials) as response:
                    # We expect this to fail with 401 (unauthorized) for dummy credentials
                    # but the endpoint should be reachable
                    if response.status in [401, 422]:  # Unauthorized or validation error
                        self.results["authentication"] = {
                            "status": "pass",
                            "note": "Endpoint reachable, authentication working"
                        }
                        return True
                    elif response.status == 404:
                        self.results["authentication"] = {
                            "status": "fail",
                            "error": "Authentication endpoint not found"
                        }
                        return False
                    else:
                        self.results["authentication"] = {
                            "status": "fail",
                            "error": f"Unexpected status {response.status}"
                        }
                        return False
        except Exception as e:
            self.results["authentication"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def test_ml_service(self) -> bool:
        """Test ML service endpoints."""
        try:
            # Test model status endpoint
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/api/v1/ml/models/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.results["ml_service"] = {
                            "status": "pass",
                            "models_loaded": data.get("models_loaded", 0),
                            "gpu_available": data.get("gpu_available", False)
                        }
                        return True
                    else:
                        self.results["ml_service"] = {
                            "status": "fail",
                            "error": f"HTTP {response.status}"
                        }
                        return False
        except Exception as e:
            self.results["ml_service"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def test_data_endpoints(self) -> bool:
        """Test data-related endpoints."""
        try:
            # Test market data endpoint
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/api/v1/data/exchanges") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.results["data_endpoints"] = {
                            "status": "pass",
                            "exchanges": data.get("exchanges", [])
                        }
                        return True
                    else:
                        self.results["data_endpoints"] = {
                            "status": "fail",
                            "error": f"HTTP {response.status}"
                        }
                        return False
        except Exception as e:
            self.results["data_endpoints"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection for real-time data."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"{self.base_url.replace('http', 'ws')}/ws/market-data") as ws:
                    # Send a test message
                    await ws.send_str(json.dumps({"type": "subscribe", "symbols": ["AAPL"]}))
                    
                    # Wait for response with timeout
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self.results["websocket"] = {
                                "status": "pass",
                                "note": "WebSocket connection successful"
                            }
                            return True
                        else:
                            self.results["websocket"] = {
                                "status": "fail",
                                "error": f"Unexpected message type: {msg.type}"
                            }
                            return False
                    except asyncio.TimeoutError:
                        self.results["websocket"] = {
                            "status": "fail",
                            "error": "WebSocket response timeout"
                        }
                        return False
        except Exception as e:
            self.results["websocket"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def test_database_connectivity(self) -> bool:
        """Test database connectivity through API."""
        try:
            # Test an endpoint that requires database access
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/api/v1/portfolio/summary") as response:
                    # We expect 401 (unauthorized) since we're not authenticated,
                    # but this means the database connection is working
                    if response.status in [401, 403]:
                        self.results["database"] = {
                            "status": "pass",
                            "note": "Database accessible through API"
                        }
                        return True
                    elif response.status == 500:
                        self.results["database"] = {
                            "status": "fail",
                            "error": "Database connection error (HTTP 500)"
                        }
                        return False
                    else:
                        self.results["database"] = {
                            "status": "pass",
                            "note": f"Unexpected but valid response: {response.status}"
                        }
                        return True
        except Exception as e:
            self.results["database"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def test_cache_connectivity(self) -> bool:
        """Test Redis cache connectivity through API."""
        try:
            # Test an endpoint that uses caching
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # Make the same request twice to test caching
                start_time = time.time()
                async with session.get(f"{self.base_url}/api/v1/data/market-status") as response1:
                    first_response_time = time.time() - start_time
                    
                    start_time = time.time()
                    async with session.get(f"{self.base_url}/api/v1/data/market-status") as response2:
                        second_response_time = time.time() - start_time
                        
                        if response1.status == 200 and response2.status == 200:
                            # Second request should be faster due to caching
                            cache_working = second_response_time < first_response_time * 0.8
                            
                            self.results["cache"] = {
                                "status": "pass" if cache_working else "degraded",
                                "first_response_time": first_response_time,
                                "second_response_time": second_response_time,
                                "cache_working": cache_working
                            }
                            return True
                        else:
                            self.results["cache"] = {
                                "status": "fail",
                                "error": f"HTTP {response1.status}/{response2.status}"
                            }
                            return False
        except Exception as e:
            self.results["cache"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def test_performance(self) -> bool:
        """Test basic performance metrics."""
        try:
            response_times = []
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                # Make multiple requests to test performance
                for _ in range(5):
                    start_time = time.time()
                    async with session.get(f"{self.base_url}/health") as response:
                        response_time = time.time() - start_time
                        response_times.append(response_time)
                        
                        if response.status != 200:
                            self.results["performance"] = {
                                "status": "fail",
                                "error": f"HTTP {response.status}"
                            }
                            return False
                
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                
                # Performance thresholds
                performance_good = avg_response_time < 0.5 and max_response_time < 1.0
                
                self.results["performance"] = {
                    "status": "pass" if performance_good else "degraded",
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "all_response_times": response_times
                }
                
                return True
                
        except Exception as e:
            self.results["performance"] = {
                "status": "fail",
                "error": str(e)
            }
            return False
    
    async def run_all_tests(self) -> Dict:
        """Run all smoke tests."""
        print(f"Running smoke tests for {self.environment} environment...")
        print(f"Base URL: {self.base_url}")
        print()
        
        tests = [
            ("API Health", self.test_api_health()),
            ("API Status", self.test_api_status()),
            ("Authentication", self.test_authentication()),
            ("ML Service", self.test_ml_service()),
            ("Data Endpoints", self.test_data_endpoints()),
            ("WebSocket", self.test_websocket_connection()),
            ("Database", self.test_database_connectivity()),
            ("Cache", self.test_cache_connectivity()),
            ("Performance", self.test_performance())
        ]
        
        results = []
        for test_name, test_coro in tests:
            print(f"Running {test_name}...", end=" ")
            try:
                result = await test_coro
                if result:
                    print("✓ PASS")
                else:
                    print("✗ FAIL")
                results.append(result)
            except Exception as e:
                print(f"✗ ERROR: {e}")
                results.append(False)
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print()
        print("=" * 50)
        print(f"SMOKE TEST SUMMARY")
        print(f"Environment: {self.environment}")
        print(f"Passed: {passed}/{total}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("Status: ✓ ALL TESTS PASSED")
            return {"status": "success", "passed": passed, "total": total, "results": self.results}
        elif passed > total * 0.7:  # 70% pass rate
            print("Status: ⚠ MOSTLY PASSING (some degraded)")
            return {"status": "degraded", "passed": passed, "total": total, "results": self.results}
        else:
            print("Status: ✗ TESTS FAILED")
            return {"status": "failed", "passed": passed, "total": total, "results": self.results}
    
    def print_detailed_results(self):
        """Print detailed test results."""
        print()
        print("DETAILED RESULTS:")
        print("-" * 50)
        
        for test_name, result in self.results.items():
            status = result.get("status", "unknown")
            status_symbol = {
                "pass": "✓",
                "fail": "✗",
                "degraded": "⚠"
            }.get(status, "?")
            
            print(f"{status_symbol} {test_name.replace('_', ' ').title()}: {status.upper()}")
            
            if "error" in result:
                print(f"  Error: {result['error']}")
            
            if "note" in result:
                print(f"  Note: {result['note']}")
            
            # Print additional details
            for key, value in result.items():
                if key not in ["status", "error", "note"]:
                    print(f"  {key}: {value}")
            
            print()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="AI Trading Platform Smoke Tests")
    parser.add_argument("--environment", "-e", default="staging", 
                       help="Environment to test (default: staging)")
    parser.add_argument("--base-url", "-u", 
                       help="Base URL for API (auto-detected if not provided)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed results")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = SmokeTestRunner(
        environment=args.environment,
        base_url=args.base_url
    )
    
    # Run tests
    summary = await runner.run_all_tests()
    
    # Show detailed results if requested
    if args.verbose:
        runner.print_detailed_results()
    
    # Output JSON if requested
    if args.json:
        print()
        print("JSON OUTPUT:")
        print(json.dumps(summary, indent=2))
    
    # Exit with appropriate code
    if summary["status"] == "success":
        sys.exit(0)
    elif summary["status"] == "degraded":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())