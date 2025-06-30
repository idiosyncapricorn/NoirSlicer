import threading
import core_cpp

def python_task(n):
    print(f"[Python] computing 10×{n}…", end=" ")
    return n * 10

def main():
    barrier = threading.Barrier(2)

    def run_cpp():
        barrier.wait()   # wait for both threads
        res = core_cpp.heavy_compute(7)
        print("[C++] result:", res)

    def run_py():
        barrier.wait()
        res = python_task(7)
        print("[Python] result:", res)

    t_cpp = threading.Thread(target=run_cpp)
    t_py  = threading.Thread(target=run_py)
    t_cpp.start()
    t_py.start()
    t_cpp.join()
    t_py.join()

if __name__ == "__main__":
    main()
