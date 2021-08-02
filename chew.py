#!/usr/local/bin/python3
# trying to heat cpus
count_of_primes = 0
for i in range(1000 * 1000 * 1000):
    prime = True
    for j in range(2, i):
        if i % j == 0:
            prime = False
            break
    if prime:
        count_of_primes += 1
