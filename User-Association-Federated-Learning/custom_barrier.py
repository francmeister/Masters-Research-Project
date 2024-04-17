import threading

class CustomBarrier:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count = 0
        self.condition = threading.Condition()

    def wait_for_aggregation(self, global_entity, local_model, access_point_number):
        with self.condition:
            self.count += 1
            if self.count == self.num_threads:
                # All threads have reached the aggregation point
                # Perform the aggregation here
                print("Performing aggregation. Round: ", global_entity.rounds)
                global_entity.aggregate_local_models()
                # Reset the count for the next iteration
                self.count = 0
                # Notify all threads that aggregation is complete
                self.condition.notify_all()
            else:
                # Wait for aggregation to complete
                print("Access Point: ", access_point_number, " waiting for model aggregation")
                global_entity.acquire_local_model(local_model)
                self.condition.wait()