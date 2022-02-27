import sys

sys.stdout = open("test.txt", "w")

print("Hello World")

sys.stdout.close()