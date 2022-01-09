import numpy as np


def main():
    xs = [3, 7, 6] # Assuming we don't know length

    mean = 0  # xs[0]
    #i = 1
    for i in range(len(xs)):
      mean = mean*i/(i+1) + xs[i]/(i+1)
    print(mean)
    print(np.array(xs).mean())


if __name__ == '__main__':
    begin = np.array([20, 21, 26, 25])
    for j in range(3):
        for i in range(4):
            a = begin + i
            print(f"            ({a[0]} {a[1]} {a[2]} {a[3]})")
        begin += 5
    # main()
