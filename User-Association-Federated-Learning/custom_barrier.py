import threading
import numpy as np

class CustomBarrier:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        print('self.num_threads')
        print(self.num_threads)
        self.count = 0
        self.condition = threading.Condition()
        self.local_associations = []

    def wait_for_aggregation(self, global_entity, local_model, access_point_number):
        with self.condition:
            self.count += 1
            if self.count == self.num_threads:
                # All threads have reached the aggregation point
                # Perform the aggregation here
                global_entity.acquire_local_model(local_model)
                print("Performing model aggregation. Round: ", global_entity.rounds)
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

                
    def wait_for_reassociations(self, env, global_entity, access_point_number, episode_reward,access_point_radius):
        with self.condition:
            self.count += 1
            if self.count == self.num_threads:
                # All threads have reached the aggregation point
                # Perform the aggregation here
                SBS_association = env.SBS.predict_future_association(access_point_radius)
                self.local_associations.append(SBS_association)
                global_entity.acquire_local_user_associations(SBS_association)
                global_entity.calculate_global_reward(episode_reward)

                user_association = global_entity.aggregate_user_associations()
                self.local_associations = np.array(self.local_associations)
                print('self.local_associations')
                print(self.local_associations)
                print('aggregated user_association')
                print(user_association)
                self.local_associations = []
                env.SBS.reassociate_users(user_association)
                #env.SBS.reassociate_users(np.array([1,2,3,3,2,1,2,2,3,1,3,2]))
                env.SBS.populate_buffer_memory_sample_with_reward(global_entity.global_reward)
                # Reset the count for the next iteration
                self.count = 0
                # Notify all threads that aggregation is complete
                self.condition.notify_all()
            else:
                # Wait for aggregation to complete
                print("Access Point: ", access_point_number, " waiting for reassociations")
                SBS_association = env.SBS.predict_future_association(access_point_radius)
                self.local_associations.append(SBS_association)
                global_entity.acquire_local_user_associations(SBS_association)
                global_entity.calculate_global_reward(episode_reward)
                self.condition.wait()
