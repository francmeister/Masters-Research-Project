import threading
import numpy as np
import copy

class CustomBarrier:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count = 0
        self.condition = threading.Condition()
        self.local_associations = []
        self.reassociations = []

    def wait_for_aggregation(self, global_entity, local_model, access_point_number):
        with self.condition:
            self.count += 1
            if self.count == self.num_threads:
                # All threads have reached the aggregation point
                # Perform the aggregation here
                global_entity.acquire_local_model(local_model)
                #print("Performing model aggregation. Round: ", global_entity.rounds)
                global_entity.aggregate_local_models()
                # Reset the count for the next iteration
                self.count = 0
                # Notify all threads that aggregation is complete
                self.condition.notify_all()
            else:
                # Wait for aggregation to complete
                #print("Access Point: ", access_point_number, " waiting for model aggregation")
                global_entity.acquire_local_model(local_model)
                self.condition.wait()

                
    def wait_for_reassociations(self, env, global_entity, timestep_counter, episode_reward,access_point_radius):
        with self.condition:
            self.count += 1
            if self.count == self.num_threads:
                # All threads have reached the aggregation point
                # Perform the aggregation here
                env.SBS.acquire_global_model(global_entity.global_model)
                if timestep_counter <= 50000:
                    global_entity_random_associations = global_entity.perform_random_association(env.SBS.all_users)
                    SBS_association = env.SBS.random_based_association(global_entity_random_associations,access_point_radius, timestep_counter, env.eMBB_Users, env.URLLC_Users)
                else:
                    SBS_association = env.SBS.predict_future_association(access_point_radius, timestep_counter, env.eMBB_Users, env.URLLC_Users)
                self.local_associations.append(SBS_association)
                global_entity.acquire_local_user_associations(SBS_association)
                global_entity.calculate_global_reward(episode_reward)
                np.save("./results/", global_entity.global_reward)

                #user_association = global_entity.aggregate_user_associations()
                self.local_associations = np.array(self.local_associations)
                # print('self.local_associations')
                # print(self.local_associations)
                #print('aggregated user_association')
                #print(user_association)
                self.local_associations = []
                # print('env.SBS.SBS_label')
                # print(env.SBS.SBS_label)
                self.reassociations.clear()
               # self.reassociations = copy.deepcopy(user_association)
                #env.SBS.reassociate_users(user_association)
                #env.SBS.reassociate_users(np.array([1,2,3,3,2,1,2,2,3,1,3,2]))
                #env.SBS.populate_buffer_memory_sample_with_reward(global_entity.global_reward)
                # Reset the count for the next iteration
                self.count = 0
                # Notify all threads that aggregation is complete
                self.condition.notify_all()
            else:
                # Wait for aggregation to complete
                #print("Access Point: ", access_point_number, " waiting for reassociations")
                env.SBS.acquire_global_model(global_entity.global_model)
                if timestep_counter <= 50000:
                    global_entity_random_associations = global_entity.perform_random_association(env.SBS.all_users)
                    SBS_association = env.SBS.random_based_association(global_entity_random_associations,access_point_radius, timestep_counter, env.eMBB_Users, env.URLLC_Users)
                else:
                    SBS_association = env.SBS.predict_future_association(access_point_radius, timestep_counter, env.eMBB_Users, env.URLLC_Users)
                self.local_associations.append(SBS_association)
                global_entity.acquire_local_user_associations(SBS_association)
                global_entity.calculate_global_reward(episode_reward)
                #print('env.SBS.SBS_label: ', env.SBS.SBS_label)
                self.condition.wait()
