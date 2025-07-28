import numpy as np
import pandas as pd
import random
import time
import uuid
from typing import List, Dict
import json

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class NormalizedDevicePriorityGenerator:
    """Generate normalized synthetic dataset for device priority ranking"""
    
    def __init__(self):
        # Define categorical options based on your requirements
        self.device_types = [
            'Desktop/Laptop', 'Mobile/Tablet', 'Smart TV', 
            'IoT Device', 'Security Camera', 'Gaming Console'
        ]
        
        self.device_names = [
            'Mac', 'Windows PC', 'Linux Workstation', 'MacBook Pro', 'ThinkPad',
            'iPhone', 'iPad', 'Samsung Galaxy', 'Android Tablet',
            'Samsung TV', 'LG TV', 'Sony TV', 'Apple TV',
            'Smart Bulb', 'Smart Thermostat', 'Smart Speaker', 'Smart Lock',
            'Security Cam 1', 'Security Cam 2', 'Doorbell Camera',
            'PlayStation 5', 'Xbox Series X', 'Nintendo Switch'
        ]
        
        self.locations = ['bedroom', 'living_room', 'kitchen', 'office', 'garage', 'basement']
        self.frequencies = ['2.4G', '5G']  # Removed 6G as requested
        self.bandwidth_options = ['20MHz', '40MHz', '80MHz', '160MHz']
        
        # Priority scoring weights (normalized)
        self.priority_weights = {
            'device_type_scores': {
                'Desktop/Laptop': 0.9,
                'Mobile/Tablet': 0.7,
                'Gaming Console': 0.8,
                'Security Camera': 0.8,
                'Smart TV': 0.6,
                'IoT Device': 0.3
            },
            'location_scores': {
                'office': 0.9,
                'bedroom': 0.7,
                'living_room': 0.6,
                'kitchen': 0.5,
                'garage': 0.4,
                'basement': 0.3
            },
            'frequency_scores': {
                '5G': 0.8,
                '2.4G': 0.4
            },
            'bandwidth_scores': {
                '160MHz': 0.9,
                '80MHz': 0.7,
                '40MHz': 0.5,
                '20MHz': 0.3
            }
        }
        
    def generate_device_id(self) -> str:
        """Generate unique device ID"""
        return str(uuid.uuid4()).replace('-', '')
    
    def generate_mac_address(self) -> str:
        """Generate realistic MAC address"""
        return ':'.join([f"{random.randint(0, 255):02x}" for _ in range(6)])
    
    def generate_timestamp(self) -> int:
        """Generate realistic timestamp (current time range)"""
        # Generate timestamp for last 30 days
        base_time = int(time.time() * 1000)  # Current time in milliseconds
        random_offset = random.randint(0, 30 * 24 * 60 * 60 * 1000)  # 30 days in ms
        return base_time - random_offset
    
    def calculate_normalized_priority(self, device_data: Dict) -> float:
        """Calculate normalized priority score (0-1) based on multiple factors"""
        priority_score = 0.0
        
        # Device type influence (30%)
        device_type_score = self.priority_weights['device_type_scores'].get(
            device_data['deviceType'], 0.5
        )
        priority_score += device_type_score * 0.30
        
        # Location influence (15%)
        location_score = self.priority_weights['location_scores'].get(
            device_data['deviceLocation'], 0.5
        )
        priority_score += location_score * 0.15
        
        # Network quality influence (25%)
        # MOS Score (higher is better)
        mos_normalized = device_data['deviceMOSScore'] / 5.0
        priority_score += mos_normalized * 0.10
        
        # Latency (lower is better) - normalize to 0-1 where 0ms=1.0, 100ms=0.0
        latency_normalized = max(0, 1.0 - (device_data['deviceLatency'] / 100.0))
        priority_score += latency_normalized * 0.08
        
        # Packet Loss (lower is better) - normalize where 0%=1.0, 10%=0.0
        packet_loss_normalized = max(0, 1.0 - (device_data['devicePacketLoss'] / 10.0))
        priority_score += packet_loss_normalized * 0.07
        
        # Signal strength influence (10%)
        # RSSI (higher is better) - normalize where -30dBm=1.0, -80dBm=0.0
        rssi_normalized = max(0, (device_data['rssi'] + 80) / 50.0)
        priority_score += rssi_normalized * 0.10
        
        # Bandwidth and frequency influence (15%)
        freq_score = self.priority_weights['frequency_scores'].get(
            device_data['bandwidthFreq'], 0.5
        )
        bandwidth_score = self.priority_weights['bandwidth_scores'].get(
            device_data['currentBandwidth'], 0.5
        )
        priority_score += (freq_score + bandwidth_score) / 2 * 0.15
        
        # Time-based influence (5%)
        # Peak hours (9-17 weekdays) get higher priority
        is_peak_hour = (
            device_data['hourOfTheDay'] >= 9 and 
            device_data['hourOfTheDay'] <= 17 and 
            device_data['dayOfTheWeek'] < 5  # Monday=0 to Friday=4
        )
        time_score = 0.8 if is_peak_hour else 0.5
        priority_score += time_score * 0.05
        
        # Add small random variation for realism
        priority_score += random.uniform(-0.02, 0.02)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, priority_score))
    
    def generate_single_device(self) -> Dict:
        """Generate single normalized device data"""
        device_type = random.choice(self.device_types)
        
        # Select device name based on type
        if device_type == 'Desktop/Laptop':
            device_name = random.choice(['Mac', 'Windows PC', 'Linux Workstation', 'MacBook Pro', 'ThinkPad'])
        elif device_type == 'Mobile/Tablet':
            device_name = random.choice(['iPhone', 'iPad', 'Samsung Galaxy', 'Android Tablet'])
        elif device_type == 'Smart TV':
            device_name = random.choice(['Samsung TV', 'LG TV', 'Sony TV', 'Apple TV'])
        elif device_type == 'Gaming Console':
            device_name = random.choice(['PlayStation 5', 'Xbox Series X', 'Nintendo Switch'])
        elif device_type == 'Security Camera':
            device_name = random.choice(['Security Cam 1', 'Security Cam 2', 'Doorbell Camera'])
        else:  # IoT Device
            device_name = random.choice(['Smart Bulb', 'Smart Thermostat', 'Smart Speaker', 'Smart Lock'])
        
        # Generate time features
        hour_of_day = random.randint(0, 23)
        day_of_week = random.randint(0, 6)  # 0=Monday, 6=Sunday
        
        # Generate network quality metrics based on device type (as integers)
        if device_type == 'Desktop/Laptop':
            device_latency = random.randint(1, 20)
            device_jitter = random.randint(0, 5)
            device_packet_loss = random.randint(0, 2)
            device_mos_score = random.randint(35, 50) / 10.0  # 3.5 to 5.0
            rssi = random.randint(-50, -30)
        elif device_type == 'Gaming Console':
            device_latency = random.randint(1, 15)
            device_jitter = random.randint(0, 3)
            device_packet_loss = random.randint(0, 1)
            device_mos_score = random.randint(40, 50) / 10.0  # 4.0 to 5.0
            rssi = random.randint(-45, -30)
        elif device_type == 'Mobile/Tablet':
            device_latency = random.randint(5, 50)
            device_jitter = random.randint(1, 10)
            device_packet_loss = random.randint(0, 5)
            device_mos_score = random.randint(30, 45) / 10.0  # 3.0 to 4.5
            rssi = random.randint(-70, -40)
        elif device_type == 'Smart TV':
            device_latency = random.randint(10, 40)
            device_jitter = random.randint(2, 8)
            device_packet_loss = random.randint(0, 3)
            device_mos_score = random.randint(35, 45) / 10.0  # 3.5 to 4.5
            rssi = random.randint(-60, -35)
        elif device_type == 'Security Camera':
            device_latency = random.randint(5, 30)
            device_jitter = random.randint(1, 6)
            device_packet_loss = random.randint(0, 2)
            device_mos_score = random.randint(30, 40) / 10.0  # 3.0 to 4.0
            rssi = random.randint(-65, -40)
        else:  # IoT Device
            device_latency = random.randint(10, 100)
            device_jitter = random.randint(2, 15)
            device_packet_loss = random.randint(0, 8)
            device_mos_score = random.randint(20, 35) / 10.0  # 2.0 to 3.5
            rssi = random.randint(-80, -50)
        
        # Generate device data - ONLY the features you specified
        device_data = {
            'id': self.generate_device_id(),
            'timestamp': self.generate_timestamp(),
            'hourOfTheDay': hour_of_day,
            'dayOfTheWeek': day_of_week,
            'deviceType': device_type,
            'deviceName': device_name,
            'mac': self.generate_mac_address(),
            'deviceLatency': device_latency,
            'deviceJitter': device_jitter,
            'devicePacketLoss': device_packet_loss,
            'deviceMOSScore': device_mos_score,
            'deviceLocation': random.choice(self.locations),
            'rssi': rssi,
            'currentBandwidth': random.choice(self.bandwidth_options),
            'bandwidthFreq': random.choice(self.frequencies)
        }
        
        return device_data
    
    def generate_session_data(self, num_devices: int = 5) -> List[Dict]:
        """Generate data for multiple devices (no session concept)"""
        devices = []
        
        for _ in range(num_devices):
            device = self.generate_single_device()
            devices.append(device)
        
        return devices
    
    def generate_dataset(self, num_devices: int = 10000) -> pd.DataFrame:
        """Generate complete dataset with only your specified features"""
        print(f"Generating {num_devices} device records...")
        
        all_data = []
        
        for i in range(num_devices):
            if i % 1000 == 0:
                print(f"Generated {i}/{num_devices} devices...")
            
            device = self.generate_single_device()
            all_data.append(device)
        
        df = pd.DataFrame(all_data)
        return df
    
    def validate_dataset(self, df: pd.DataFrame):
        """Validate the generated dataset for quality"""
        print(f"\n=== Dataset Validation ===")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("✓ No missing values found")
        else:
            print(f"✗ Missing values detected: {missing_values[missing_values > 0].to_dict()}")
        
        # Check data ranges
        validation_checks = [
            ('deviceLatency', 0, 200),
            ('deviceJitter', 0, 20),
            ('devicePacketLoss', 0, 15),
            ('deviceMOSScore', 1, 5),
            ('rssi', -100, -10),
            ('hourOfTheDay', 0, 23),
            ('dayOfTheWeek', 0, 6)
        ]
        
        for column, min_val, max_val in validation_checks:
            actual_min = df[column].min()
            actual_max = df[column].max()
            if actual_min >= min_val and actual_max <= max_val:
                print(f"✓ {column}: [{actual_min:.2f}, {actual_max:.2f}] within expected range")
            else:
                print(f"✗ {column}: [{actual_min:.2f}, {actual_max:.2f}] outside expected range [{min_val}, {max_val}]")
        
        # Check categorical distributions
        print(f"\n=== Categorical Distributions ===")
        categorical_columns = ['deviceType', 'deviceLocation', 'bandwidthFreq', 'currentBandwidth']
        for col in categorical_columns:
            distribution = df[col].value_counts()
            print(f"{col}: {dict(distribution)}")
    
    def save_dataset(self, df: pd.DataFrame, filepath: str = 'normalized_device_priority_dataset.csv'):
        """Save dataset to CSV file with comprehensive statistics"""
        df.to_csv(filepath, index=False)
        print(f"\n=== Dataset Saved ===")
        print(f"File: {filepath}")
        print(f"Size: {len(df)} records")
        
        # Comprehensive statistics
        print(f"\n=== Dataset Statistics ===")
        print(f"Total records: {len(df):,}")
        print(f"Unique devices: {df['id'].nunique():,}")
        
        # Feature correlations - remove priority score references
        print(f"\n=== Data Distribution ===")
        numeric_cols = ['deviceLatency', 'deviceJitter', 'devicePacketLoss', 
                       'deviceMOSScore', 'rssi', 'hourOfTheDay', 'dayOfTheWeek']
        
        for col in numeric_cols:
            print(f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}")
        
        return filepath

def main():
    """Main function to generate normalized dataset"""
    print("=== Normalized Device Priority Dataset Generator ===\n")
    
    # Initialize generator
    generator = NormalizedDevicePriorityGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(num_devices=10000)
    
    # Validate dataset
    generator.validate_dataset(df)
    
    # Save dataset
    filepath = generator.save_dataset(df, 'device_priority_dataset.csv')
    
    # Display sample records
    print(f"\n=== Sample Records ===")
    sample_df = df.head(10)[['id', 'deviceType', 'deviceName', 'deviceLocation', 
                            'deviceLatency', 'deviceMOSScore', 'rssi']]
    print(sample_df.to_string(index=False))
    
    # Export feature schema
    schema = {
        'features': {
            'id': 'string - Unique device identifier',
            'timestamp': 'integer - Unix timestamp in milliseconds',
            'hourOfTheDay': 'integer - Hour (0-23)',
            'dayOfTheWeek': 'integer - Day (0=Monday, 6=Sunday)',
            'deviceType': 'categorical - Device category',
            'deviceName': 'categorical - Device model/name',
            'mac': 'string - MAC address',
            'deviceLatency': 'integer - Network latency in ms',
            'deviceJitter': 'integer - Network jitter in ms',
            'devicePacketLoss': 'integer - Packet loss percentage',
            'deviceMOSScore': 'float - Mean Opinion Score (1-5)',
            'deviceLocation': 'categorical - Physical location',
            'rssi': 'integer - Signal strength in dBm',
            'currentBandwidth': 'categorical - Bandwidth allocation',
            'bandwidthFreq': 'categorical - Frequency band (2.4G, 5G)'
        },
        'total_features': len(df.columns),
        'categorical_features': ['deviceType', 'deviceName', 'deviceLocation', 'currentBandwidth', 'bandwidthFreq'],
        'numerical_features': ['timestamp', 'hourOfTheDay', 'dayOfTheWeek', 'deviceLatency', 
                              'deviceJitter', 'devicePacketLoss', 'deviceMOSScore', 'rssi'],
        'metadata_features': ['id', 'mac']
    }
    
    with open('dataset_schema.json', 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"\n=== Schema saved to dataset_schema.json ===")
    print(f"Dataset is ready for device prioritization model training!")
    
    return df

if __name__ == "__main__":
    dataset = main()