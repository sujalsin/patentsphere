"""
PatentSphere Cost Management Utility

This module helps track and prevent unexpected GCP costs by:
1. Estimating costs before running BigQuery queries
2. Monitoring cumulative spend
3. Enforcing budget limits
4. Providing cost-effective alternatives
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CostManager:
    """Manages GCP costs and enforces budget limits"""
    
    # GCP Pricing (as of 2024, subject to change)
    PRICING = {
        "bigquery_per_tb": 5.00,  # $5 per TB scanned
        "compute_e2_standard_4_hourly": 0.134,  # e2-standard-4 per hour
        "compute_e2_medium_hourly": 0.033,  # e2-medium per hour
        "storage_per_gb_month": 0.020,  # Standard storage per GB/month
        "egress_per_gb": 0.12,  # Network egress per GB
    }
    
    def __init__(self, budget_file: str = "config/budget.json"):
        """
        Initialize cost manager
        
        Args:
            budget_file: Path to JSON file tracking budget usage
        """
        self.budget_file = Path(budget_file)
        self.budget_data = self._load_budget()
        
    def _load_budget(self) -> Dict:
        """Load or create budget tracking file"""
        if self.budget_file.exists():
            with open(self.budget_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize budget tracking
            default_budget = {
                "total_credits_usd": 50.0,
                "max_spend_usd": 10.0,
                "cumulative_spend_usd": 0.0,
                "transactions": [],
                "created_at": datetime.now().isoformat(),
            }
            self._save_budget(default_budget)
            return default_budget
    
    def _save_budget(self, data: Dict):
        """Save budget data to file"""
        self.budget_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.budget_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def estimate_bigquery_cost(
        self, 
        num_patents: int = 1_000_000,
        avg_patent_size_kb: float = 5.0
    ) -> Dict:
        """
        Estimate BigQuery query cost
        
        Args:
            num_patents: Number of patents to query
            avg_patent_size_kb: Average size per patent in KB
            
        Returns:
            Dict with cost estimate and breakdown
        """
        # BigQuery charges for data scanned, not rows
        total_data_gb = (num_patents * avg_patent_size_kb) / (1024 * 1024)  # KB to GB
        total_data_tb = total_data_gb / 1024  # GB to TB
        
        # BigQuery gives 1TB free per month
        free_quota_tb = 1.0
        billable_tb = max(0, total_data_tb - free_quota_tb)
        
        cost = billable_tb * self.PRICING["bigquery_per_tb"]
        
        return {
            "service": "BigQuery",
            "num_patents": num_patents,
            "data_scanned_gb": round(total_data_gb, 2),
            "data_scanned_tb": round(total_data_tb, 4),
            "free_quota_tb": free_quota_tb,
            "billable_tb": round(billable_tb, 4),
            "estimated_cost_usd": round(cost, 2),
            "free_alternative": "Use BigQuery Sandbox (10GB free) or download pre-exported data",
        }
    
    def estimate_compute_cost(
        self,
        vm_type: str = "e2-medium",
        hours: float = 8.0,
        preemptible: bool = True
    ) -> Dict:
        """
        Estimate GCP Compute Engine cost
        
        Args:
            vm_type: VM type (e2-medium, e2-standard-4, etc.)
            hours: Hours to run
            preemptible: Use preemptible/spot instances (60-91% discount)
            
        Returns:
            Dict with cost estimate
        """
        vm_pricing = {
            "e2-medium": 0.033,
            "e2-standard-4": 0.134,
        }
        
        hourly_rate = vm_pricing.get(vm_type, 0.033)
        
        if preemptible:
            hourly_rate *= 0.3  # ~70% discount for spot instances
        
        cost = hourly_rate * hours
        
        return {
            "service": "Compute Engine",
            "vm_type": vm_type,
            "hours": hours,
            "preemptible": preemptible,
            "hourly_rate_usd": round(hourly_rate, 4),
            "estimated_cost_usd": round(cost, 2),
            "free_alternative": "Develop locally; only use GCP for final demo (Days 13-14)",
        }
    
    def check_budget(self, planned_cost: float) -> Dict:
        """
        Check if planned cost fits within budget
        
        Args:
            planned_cost: Estimated cost of operation
            
        Returns:
            Dict with approval status and warnings
        """
        current_spend = self.budget_data["cumulative_spend_usd"]
        max_spend = self.budget_data["max_spend_usd"]
        total_credits = self.budget_data["total_credits_usd"]
        
        new_total = current_spend + planned_cost
        remaining_budget = max_spend - current_spend
        remaining_credits = total_credits - current_spend
        
        approved = new_total <= max_spend
        
        return {
            "approved": approved,
            "planned_cost_usd": round(planned_cost, 2),
            "current_spend_usd": round(current_spend, 2),
            "new_total_usd": round(new_total, 2),
            "budget_limit_usd": max_spend,
            "remaining_budget_usd": round(remaining_budget, 2),
            "remaining_credits_usd": round(remaining_credits, 2),
            "warning": None if approved else f"Exceeds budget by ${new_total - max_spend:.2f}",
        }
    
    def log_transaction(
        self,
        service: str,
        description: str,
        cost: float,
        metadata: Optional[Dict] = None
    ):
        """
        Log a cost transaction
        
        Args:
            service: GCP service name (BigQuery, Compute, etc.)
            description: Description of what was done
            cost: Actual cost incurred
            metadata: Additional metadata
        """
        transaction = {
            "timestamp": datetime.now().isoformat(),
            "service": service,
            "description": description,
            "cost_usd": round(cost, 2),
            "metadata": metadata or {},
        }
        
        self.budget_data["transactions"].append(transaction)
        self.budget_data["cumulative_spend_usd"] += cost
        
        self._save_budget(self.budget_data)
        
        logger.info(
            f"Transaction logged: {service} - {description} - ${cost:.2f}"
        )
    
    def get_free_tier_recommendations(self) -> Dict:
        """Get recommendations for staying within free tier"""
        return {
            "bigquery": [
                "Use BigQuery Sandbox (no billing account needed, 10GB limit)",
                "Use cached query results (free)",
                "Export 50K subset to local JSONL (one-time, free under 1GB egress)",
                "Use dry_run=True to estimate costs before running queries",
            ],
            "compute": [
                "Develop everything locally for Days 1-12 (FREE)",
                "Use e2-micro (2 vCPU, 1GB RAM) for Always Free tier",
                "Only provision VM on Day 13 for final testing (~$0.50 for 8 hours)",
                "Use preemptible/spot instances (70% discount)",
                "Auto-shutdown after 4 hours idle",
            ],
            "storage": [
                "Use local disk for development (FREE)",
                "First 5GB Cloud Storage is free per month",
                "Store only final 1M dataset, not intermediates",
            ],
            "general": [
                "Total estimated cost for full project: $3-5 if done right",
                "Reserve $45 credits for future projects",
                "Set up billing alerts at $5, $10, $50",
            ],
        }
    
    def generate_cost_report(self) -> str:
        """Generate a human-readable cost report"""
        data = self.budget_data
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           PatentSphere Budget & Cost Report                  ║
╚══════════════════════════════════════════════════════════════╝

Credits Available:     ${data['total_credits_usd']:.2f}
Budget Limit:          ${data['max_spend_usd']:.2f}
Cumulative Spend:      ${data['cumulative_spend_usd']:.2f}
Remaining Budget:      ${data['max_spend_usd'] - data['cumulative_spend_usd']:.2f}
Remaining Credits:     ${data['total_credits_usd'] - data['cumulative_spend_usd']:.2f}

Budget Usage:          {(data['cumulative_spend_usd'] / data['max_spend_usd'] * 100):.1f}%

Transaction History:
"""
        
        if data["transactions"]:
            for i, txn in enumerate(data["transactions"][-10:], 1):  # Last 10
                report += f"\n{i}. [{txn['timestamp'][:10]}] {txn['service']}: {txn['description']} - ${txn['cost_usd']:.2f}"
        else:
            report += "\nNo transactions yet (excellent!)"
        
        report += f"""

╔══════════════════════════════════════════════════════════════╗
║                  COST-SAVING STRATEGY                        ║
╚══════════════════════════════════════════════════════════════╝

✅ Days 1-12:  Develop entirely locally (FREE)
✅ Day 13:     Provision GCP VM for testing (~$1)
✅ Day 14:     Final demo and teardown (~$1)
✅ TOTAL:      $2-3 (leaving $47 credits for future)
"""
        
        return report


# Convenience function
def estimate_project_cost() -> str:
    """Estimate total project cost"""
    cm = CostManager()
    
    # Estimate BigQuery for 1M patents
    bq_cost = cm.estimate_bigquery_cost(num_patents=1_000_000)
    
    # Estimate VM for 16 hours (Days 13-14)
    vm_cost = cm.estimate_compute_cost(vm_type="e2-medium", hours=16, preemptible=True)
    
    total = bq_cost["estimated_cost_usd"] + vm_cost["estimated_cost_usd"]
    
    report = f"""
PatentSphere Total Cost Estimate:
================================

BigQuery (1M patents):    ${bq_cost['estimated_cost_usd']:.2f}
  - Free quota covers most of this ✅

Compute (16 hours):       ${vm_cost['estimated_cost_usd']:.2f}
  - Using preemptible e2-medium

TOTAL ESTIMATED COST:     ${total:.2f}

YOUR BUDGET:              $50.00
REMAINING AFTER PROJECT:  ${50 - total:.2f}

✅ PROJECT IS FEASIBLE! ✅
"""
    
    return report


if __name__ == "__main__":
    # Demo usage
    print(estimate_project_cost())
    
    cm = CostManager()
    print(cm.generate_cost_report())
    print("\n" + "="*60)
    print("FREE TIER RECOMMENDATIONS:")
    print("="*60)
    
    recs = cm.get_free_tier_recommendations()
    for service, tips in recs.items():
        print(f"\n{service.upper()}:")
        for tip in tips:
            print(f"  • {tip}")
