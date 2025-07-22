import pandas as pd
import psycopg2
import time
import sys
import os
from datetime import datetime
from sqlalchemy import create_engine, inspect

class DriftMonitor:
    def __init__(self):
        self.reference_path = "/opt/airflow/monitoring/reference_data.csv"
        self.engine = create_engine('postgresql://postgres:newpassword@host.docker.internal:5432/mydb5')
        self.initialize_reference_data()
        self.reference_length = self.get_reference_length()
    
    def initialize_reference_data(self):
        """Save current table state as reference"""
        if not os.path.exists(self.reference_path):
            try:
                df = pd.read_sql('SELECT * FROM "LeadScoring"', self.engine)
                df.to_csv(self.reference_path, index=False)
                print(f"✅ Reference data saved. Initial length: {len(df)} rows")
            except Exception as e:
                print(f"❌ Failed to initialize reference data: {e}")
                sys.exit(1)

    def get_reference_length(self):
        """Get row count of reference data"""
        try:
            return len(pd.read_csv(self.reference_path))
        except Exception as e:
            print(f"❌ Failed to get reference length: {e}")
            sys.exit(1)

    def get_current_length(self):
        """Get current row count from database"""
        try:
            return pd.read_sql('SELECT COUNT(*) FROM "LeadScoring"', self.engine).iloc[0,0]
        except Exception as e:
            print(f"⚠️ Error getting current length: {e}")
            return None

    def check_drift(self):
        """Check for drift by comparing reference CSV with new Postgres data"""
        try:
            # 1. Check for new data by comparing lengths
            current_length = self.get_current_length()
            if current_length is None:
                return False

            if current_length <= self.reference_length:
                print(f"🔄 No new data (Reference: {self.reference_length}, Current: {current_length})")
                return False

            # 2. Calculate length difference
            new_records_count = current_length - self.reference_length
            diff_percentage = new_records_count / self.reference_length
            print(f"📊 New data detected: +{new_records_count} records ({diff_percentage:.1%} increase)")

            # 3. Fetch only the new records using exact row slicing
            new_data = pd.read_sql(
                f'SELECT * FROM "LeadScoring" ORDER BY id LIMIT {new_records_count} OFFSET {self.reference_length}',
                self.engine
            )

            if len(new_data) == 0:
                print("⚠️ No new data fetched despite length difference")
                return False

            # 4. Load reference data for comparison
            reference_data = pd.read_csv(self.reference_path)

            # 5. Get exact slices (no sampling)
            # Use all reference data and all new data for comparison
            reference_sample = reference_data
            new_sample = new_data

            # 6. Check for statistical drift using Evidently
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_sample, current_data=new_sample)
            results = report.as_dict()

            # 7. Analyze drift results
            drift_detected = False
            for metric in results.get('metrics', []):
                if metric['metric'] == 'DataDriftTable':
                    for feature, stats in metric['result']['drift_by_columns'].items():
                        if stats['drift_detected'] and feature not in ['id', 'timestamp']:
                            print(f"🚨 Drift detected in feature: {feature} (score: {stats['drift_score']:.2f})")
                            drift_detected = True

            # 8. Update reference data if drift found
            if drift_detected:
                print("💾 Updating reference data with new records...")
                updated_data = pd.concat([reference_data, new_data], ignore_index=True)
                updated_data.to_csv(self.reference_path, index=False)
                self.reference_length = len(updated_data)
                print(f"✅ Reference data updated. New length: {self.reference_length}")
                return True

            print("✅ No significant drift detected in new data")
            return False

        except Exception as e:
            print(f"⚠️ Drift check error: {e}")
            return False
        
    def continuous_monitoring(self):
        """Run checks every minute until drift detected"""
        print(f"🚦 Starting monitoring. Reference length: {self.reference_length} rows")
        print("🔄 Checking every minute for significant table length changes (>10%)")
        try:
            while True:
                if self.check_drift():
                    print("🚨 DRIFT_DETECTED")
                    return True
                time.sleep(60)
        except KeyboardInterrupt:
            print("🛑 Monitoring stopped by user")
            return False
        except Exception as e:
            print(f"⚠️ Monitoring error: {e}")
            return False

if __name__ == "__main__":
    try:
        monitor = DriftMonitor()
        if monitor.continuous_monitoring():
            sys.exit(42)  # Special exit code for drift detected
        sys.exit(0)
    except ImportError:
        print("❌ Required packages not installed. Please install with:")
        print("pip install sqlalchemy psycopg2-binary pandas")
        sys.exit(1)