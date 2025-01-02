import subprocess

# Path to your compiled C++ program (e.g., "program.exe" on Windows or "program" on Linux/Mac)
cpp_program_path = "./main"

for i in range(0,1000000):
    print(" STARTING TEST-SET NO.",i)
    # Run the C++ program and wait for it to finish
    process = subprocess.Popen(cpp_program_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for the process to complete
    stdout, stderr = process.communicate()
    # Print output and error (if any)
    if process.returncode == 0:
        print("C++ program output:")
        print(stdout.decode())
    else:
        print("C++ program error:")
        print(stderr.decode())

