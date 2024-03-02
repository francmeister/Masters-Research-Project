import threading
import time

def worker1():
 
    print("Worker 1 - Iteration:", 1)
    time.sleep(1)

def worker2():
    print("Worker 2 - Iteration:", 2)
    time.sleep(1)

def thread_with_threads():
    print("Outer Thread Start")

    # Spawning inner threads within the outer thread
    inner_thread1 = threading.Thread(target=worker1)
    inner_thread2 = threading.Thread(target=worker2)

    inner_thread1.start()
    inner_thread2.start()

    inner_thread1.join()
    inner_thread2.join()

    print("Outer Thread End")

def main():
    print("Main Thread Start")

    # Spawning an outer thread
    outer_thread1 = threading.Thread(target=thread_with_threads)
    outer_thread2 = threading.Thread(target=thread_with_threads)

    outer_thread1.start()
    outer_thread2.start()

    outer_thread1.join()
    outer_thread2.join()

    print("Main Thread End")

if __name__ == "__main__":
    main()
