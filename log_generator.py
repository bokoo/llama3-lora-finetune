import random
from datetime import datetime, timedelta

def random_datetime_within_30_days():
        now = datetime.now()
        random_offset = random.randint(0, 30 * 24 * 60 * 60)  # up to 30 days in seconds
        random_time = now - timedelta(seconds=random_offset)
        return random_time.strftime("%Y-%m-%d %H:%M:%S")

def save_logs_to_file(filename, logs):
    with open(filename, 'w', encoding='utf-8') as file:
        for log in logs:
            file.write(log + '\n')

class LogGenerator():
    SERVICES = [
        "Order Service", "Payment Service", "Database", "Authentication Service",
        "Message Broker", "Inventory Service", "Shipping Service", "Analytics Service",
        "Notification Service", "Logging Service", "Search Engine", "Reporting Service",
        "CRM Service"
    ]

    EVENT_EXPLANATIONS = {
        "Type1": "This is an error log indicating an internal failure occurred",
        "Type2": "This is an error log indicating an external system failure occurred",
        "Type9": "This is an error log indicating a system availability issue",
        "Type3": "This is a warning log indicating a retry attempt occurred",
        "Type4": "This is a warning log highlighting a non-critical concern",
        "Type5": "This is an info log indicating a successful operation",
        "Type6": "This is an info log indicating a successful recovery",
        "Type7": "This is an info log indicating a successful validation",
        "Type8": "This is an info log indicating a customer-related failure occurred",
        "Type10": "This is an info log indicating a data mismatch was detected",
        "Type11": "This is an info log indicating a performance check outcome",
        "Type12": "This is an info log indicating a resource usage update",
        "Type13": "This is an info log indicating a background job completed",
        "Type14": "This is an error log indicating a security violation",
        "Type15": "This is an error log indicating a request timeout occurred",
        "Type16": "This is an error log indicating an API limit breach",
        "Type17": "This is an error log indicating a critical data error",
        "Type18": "This is a warning log indicating a near-capacity alert",
        "Type19": "This is a warning log indicating deprecated usage",
        "Type20": "This is a warning log indicating an unusual traffic spike"
    }

    SEVERITY_TO_EVENT_TYPES = {
        "#Bomb": ["Type5", "Type6", "Type7", "Type8", "Type10", "Type11", "Type12", "Type13"],
        "#Candy": ["Type1", "Type2", "Type9", "Type14", "Type15", "Type16", "Type17"],
        "#Cake": ["Type3", "Type4", "Type18", "Type19", "Type20"]
    }

    def __init__(self, train_test_split_percentage = 0.8):
        self.train_test_split_percentage = train_test_split_percentage

    def create_log_entry(self):
        """
        Creates a single log entry in the format:
        [DateTime] [Severity] - ID: xxxxx - Unit: [Service] | Message: [Message] Explanation: [Explanation]
        """
        severity = random.choice(list(self.SEVERITY_TO_EVENT_TYPES.keys()))
        event_type = random.choice(self.SEVERITY_TO_EVENT_TYPES[severity])
        service = random.choice(self.SERVICES)
        log_datetime = random_datetime_within_30_days()
        log_id = random.randint(10000, 99999)
        
        message = f"{event_type} event occurred"
        explanation = f"{self.EVENT_EXPLANATIONS[event_type]} in {service}."
        
        return f"[{log_datetime}] {severity} - ID: {log_id} - Unit: {service} | Message: {message} Explanation: {explanation}"

    def generate_logs(self, num_logs=3000):
        """
        Generates 'num_logs' custom logs with the format:
        [DateTime] [Severity] - ID: xxxxx - Unit: [Service] | Message: [Message] Explanation: [Explanation]
        
        Each log is chosen from the specified severity levels, events, and services.
        The 'Message' field will mention the event type (e.g., "Type2 event occurred").
        The 'Explanation' field is a single concise sentence explaining the severity level
        and the general situation.
        """
        
        logs = [self.create_log_entry() for _ in range(num_logs)]

        # Shuffle logs to ensure randomness
        random.shuffle(logs)

        # Split dataset
        split_index = int(self.train_test_split_percentage * len(logs))
        train_logs = logs[:split_index]
        test_logs = logs[split_index:]

        # Save logs to files
        save_logs_to_file("train_logs.txt", train_logs)
        save_logs_to_file("test_logs.txt", test_logs)

generator = LogGenerator()
generator.generate_logs(1000)