import matplotlib.pyplot as plt


def save_plot(title, path, train_data, valid_data):
    figure = plt.figure(figsize=(8,6), facecolor='w')
    ax = figure.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    tline = ax.plot(train_data, linewidth=2, color='k', label='Training')
    vline = ax.plot(valid_data, linewidth=1.5, color='red', label='Validation')
    ax.legend()
    plt.savefig(path + '/' + title + '.png')