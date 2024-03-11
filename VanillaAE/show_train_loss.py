import re
import matplotlib.pyplot as plt


def main():
    # Lists to store the extracted values
    loss_pixel = []
    loss_freq = []

    # Open the text file and extract the values
    with open("C:/Users/Feng Zhunyi/Desktop/focal-frequency-loss-master/VanillaAE/experiments/celeba/logs/train_log.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r'LossPixel: (\d+\.\d+) LossFreq: (\d+\.\d+)', line)
            if match:
                loss_pixel.append(float(match.group(1)))
                loss_freq.append(float(match.group(2)))
                print(float(match.group(1)), float(match.group(2)))


    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_pixel, label="LossPixel")
    plt.plot(loss_freq, label="LossFreq")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Values")
    plt.legend()
    plt.title("LossPixel and LossFreq over Iterations")
    plt.grid(True)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()