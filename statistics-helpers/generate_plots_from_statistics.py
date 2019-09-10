#!/usr/bin/python3.7

import matplotlib
import matplotlib.pyplot as plt
import math
import json
import argparse
import numpy as np
import csv
import os
import sys

FIXED_POINT_NUMBER_FACTOR = 100000000

plot_span_length_histogram_enabled = False
plot_movie_hash_maches_enabled = False
plot_sync_state_distribution_enabled = False
plot_distance_to_reference_histogram_enabled = False
plot_offsets_by_split_penalty = True


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--statistics-dir",
    required=True,
    help="directory for statistics files (program input)",
)
parser.add_argument(
    "--plots-dir",
    required=True,
    help="directory for generated plot files (program output)",
)
parser.add_argument(
    "--file-extension", default="png", help="File extension of the generated plots"
)

args = parser.parse_args()

statistics_folder_path = args.statistics_dir
output_dir = args.plots_dir
extension = args.file_extension

os.makedirs(output_dir, exist_ok=True)

###################################################
# Span Length Histogram

with open(os.path.join(statistics_folder_path, "statistics.json"), "r") as f:
    statistics = json.load(f)


plt.rcParams.update({"figure.figsize": (7, 5), "figure.dpi": 330})


def draw_histogram(ax, bins, json_histogram, color):
    val, weight = zip(*[(int(k) / 1000.0, int(v)) for k, v in json_histogram.items()])
    height, bins = np.histogram(val, weights=weight, bins=bins)
    height = np.divide(height, np.max(height))
    ax.step(
        bins[:-1], height, "k", linestyle="-", linewidth=1, where="post", color=color
    )
    ax.bar(
        bins[:-1],
        height,
        width=np.diff(bins),
        linewidth=0,
        facecolor=color,
        alpha=0.3,
        align="edge",
    )


class OffsetStatistics:
    def __init__(self, histogram):
        occurrences = histogram["occurrences"]
        occurrences_sorted = sorted(
            [(int(offset), int(count)) for (offset, count) in occurrences.items()],
            key=lambda d: d[0],
        )
        total = sum([count for (offset, count) in occurrences_sorted])

        self.min = occurrences_sorted[0][0]
        self.max = occurrences_sorted[-1][0]

        idx = 0
        current_offset, last_bin_idx = occurrences_sorted[idx]

        self.percentiles = {}

        for percentile in range(0, 100):
            perc_idx = int((percentile / 100) * total)

            while True:
                if perc_idx < last_bin_idx:
                    self.percentiles[percentile] = current_offset
                    break
                else:
                    idx = idx + 1
                    current_offset, count = occurrences_sorted[idx]
                    last_bin_idx = last_bin_idx + count

        pass


if plot_offsets_by_split_penalty:

    plt.figure(num=None, figsize=(8, 8), dpi=200, facecolor="w", edgecolor="k")

    def plot_split_penalties(ax, statistics, histogram_name, title):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_ylim([0, 10000])
        # ax.set_yscale("log")

        plotted_percentiles = [20, 50, 80, 90, 95, 99]
        perc_heights_array = [[] for _ in range(0, len(plotted_percentiles))]
        plotted_percentiles_color = [
            "black",
            "darkred",
            "red",
            "orange",
            "green",
            "darkgreen",
        ]

        ind = []
        split_penalties = []

        data = [
            (float(split_penalty_str), histogram)
            for (split_penalty_str, histogram) in statistics[histogram_name].items()
        ]
        data = sorted(data, key=lambda v: v[0])

        for i, (split_penalty_str, histogram) in enumerate(data):
            split_penalty = float(split_penalty_str) / FIXED_POINT_NUMBER_FACTOR

            split_penalties.append(split_penalty)
            ind.append(i)

            offset_statistics = OffsetStatistics(histogram)

            for height_array, percentile in zip(
                reversed(perc_heights_array), reversed(plotted_percentiles)
            ):
                height_array.append(offset_statistics.percentiles[percentile])

        raw_offset_statistics = OffsetStatistics(statistics["raw_distance_histogram"])

        for height_array, percentile in zip(
            reversed(perc_heights_array), reversed(plotted_percentiles)
        ):
            height_array.append(raw_offset_statistics.percentiles[percentile])

        ind = list(range(0, len(data))) + [len(data) + 1]

        for (perc_heights_array, percentile, color) in zip(
            reversed(perc_heights_array),
            reversed(plotted_percentiles),
            reversed(plotted_percentiles_color),
        ):
            label = "%sth percentile" % percentile

            plt.bar(
                ind, np.array(perc_heights_array), width=0.8, label=label, color=color
            )

        plt.xticks(
            ind,
            [
                int(split_penalty) if split_penalty > 1 else split_penalty
                for split_penalty in split_penalties
            ]
            + ["raw"],
        )
        plt.ylabel("Offset in milliseconds")
        plt.xlabel("Split penalty")
        plt.legend(loc="upper right", bbox_to_anchor=(0, 0, 0.9, 1))
        plt.title(title)
        plt.xticks(rotation=45)

    ax = plt.subplot(211)
    plot_split_penalties(
        ax,
        statistics,
        "sync_to_video_offset_histogram_by_split_penalty",
        "Synchronizing to video",
    )

    ax = plt.subplot(212)
    plot_split_penalties(
        ax,
        statistics,
        "sync_to_sub_offset_histogram_by_split_penalty",
        "Synchronizing to subtitle",
    )

    plt.tight_layout(pad=0.7, h_pad=1, rect=(0, 0, 1, 0.9))

    plt.suptitle("Comparision of split penalties", fontsize=20)

    plt.savefig(os.path.join(output_dir, "split-penalties." + extension))
    # plt.show()
    plt.close()


if plot_span_length_histogram_enabled:

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.xticks(range(0, 11, 1), fontsize=14)
    plt.yticks(fontsize=14)

    binwidth = 0.05
    bins = np.arange(0, 10 + binwidth, binwidth)

    vad_color = (1, 0.5, 0.05)  # '#ff7f0e'
    vad_color_light = tuple([c * 0.3 + 0.7 for c in vad_color])
    subtitle_color = (0.12, 0.42, 0.74)  # '#1f77b4'
    subtitle_color_light = tuple([c * 0.3 + 0.7 for c in subtitle_color])

    draw_histogram(
        ax, bins, statistics["vad_span_length_histogram"]["occurrences"], vad_color
    )
    draw_histogram(
        ax,
        bins,
        statistics["subtitle_span_length_histogram"]["occurrences"],
        subtitle_color,
    )

    ax.set(
        title=None, ylabel="Normalized Frequency", xlabel="Length of spans in seconds"
    )
    ax.set_xlim([0, 10])
    ax.get_yaxis().set_ticks([])
    ax.legend(["Spans from Voice-Activity-Detection", "Spans from Subtitles"])
    # ax.set_ylim([0, 1])

    plt.savefig(os.path.join(output_dir, "span-lengths-histogram." + extension))
    # plt.show()
    plt.close()


def make_beautiful_pie_chart(ax, labels, title, sizes, explode, pie_colors):

    pie_colors = [matplotlib.colors.to_rgb(c) for c in pie_colors]
    pie_colors_light = [(r, g, b, 0.3) for (r, g, b) in pie_colors]
    pie_colors_light2 = [(r, g, b, 1) for (r, g, b) in pie_colors]

    wedges, _, _ = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        pctdistance=0.85,
        startangle=90,
        colors=pie_colors_light,
    )
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis("equal")
    ax.set(title=title)

    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    centre_circle.set_linewidth(2)
    centre_circle.set_edgecolor("#444444")
    ax.add_artist(centre_circle)

    for w, edge_color in zip(wedges, pie_colors_light2):
        w.set_linewidth(2)
        w.set_edgecolor(edge_color)

    pass


if plot_movie_hash_maches_enabled:

    labels = ["No match", "Match"]
    sizes = [
        statistics["general"]["total_movie_count"]
        - statistics["general"]["movie_with_ref_sub_count"],
        statistics["general"]["movie_with_ref_sub_count"],
    ]
    explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    pie_colors = ["red", "green"]

    fig, ax = plt.subplots()
    make_beautiful_pie_chart(
        ax,
        labels,
        "Movie file hash matches on 'OpenSubtitles.org'",
        sizes,
        explode,
        pie_colors,
    )

    plt.savefig(os.path.join(output_dir, "movie-hash-matches." + extension))
    # plt.show()
    plt.close()


def plot_sync_class_distribution(ax, input_data, title):
    unknown_count = input_data["unknown"]
    synced_count = input_data["synced"]
    unsynced_count = input_data["unsynced"]

    labels = ["Unkown", "Synchronized", "Unsynchronized"]
    explode = (0, 0, 0)
    pie_colors = ["grey", "green", "red"]
    sizes = [unknown_count, synced_count, unsynced_count]

    make_beautiful_pie_chart(ax, labels, title, sizes, explode, pie_colors)


if plot_sync_state_distribution_enabled:

    fig = plt.gcf()
    fig.suptitle("Percentage of synchronized subtitles", fontsize=16)

    ax = plt.subplot(221)
    plot_sync_class_distribution(
        ax, statistics["general"]["raw_sync_class_counts"], title="Raw subtitle files"
    )

    ax = plt.subplot(222)
    plot_sync_class_distribution(
        ax,
        statistics["general"]["sync_to_video_sync_class_counts"],
        title="Aligned to video",
    )

    ax = plt.subplot(223)
    plot_sync_class_distribution(
        ax,
        statistics["general"]["sync_to_sub_sync_class_counts"],
        title="Aligned to subtitle",
    )

    plt.savefig(os.path.join(output_dir, "sync-state-distribution." + extension))
    # plt.show()
    plt.close()

if plot_distance_to_reference_histogram_enabled:

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    startrange = 0.01
    endrange = 30.0
    # plt.xticks([startrange, endrange], fontsize=14)
    # plt.yticks(fontsize=14)

    binwidth = 0.05
    bins = np.logspace(np.log10(startrange), np.log10(endrange), 300)  #

    def plot_cusum(json_histogram, bins, ax, color):

        items = [(int(k), int(v)) for k, v in json_histogram["occurrences"].items()]
        count_sum = sum([count for (ms, count) in items])

        items = [
            (ms / 1000.0, count) if ms > 10 else ((10 / 1000.0), count)
            for (ms, count) in items
        ]
        val, weight = zip(*items)
        height, bins = np.histogram(val, weights=weight, bins=bins)
        height = np.cumsum(height)
        ax.step(
            bins[:-1],
            height,
            "k",
            linestyle="-",
            linewidth=1,
            where="post",
            color=color,
        )
        ax.bar(
            bins[:-1],
            height,
            width=np.diff(bins),
            linewidth=0,
            facecolor=color,
            alpha=0.3,
            align="edge",
        )

        return count_sum

    count_sum1 = plot_cusum(statistics["raw_distance_histogram"], bins, ax, "red")

    count_sum2 = plot_cusum(
        statistics["sync_to_video_distance_histogram"], bins, ax, "orange"
    )
    assert count_sum1 == count_sum2

    count_sum3 = plot_cusum(
        statistics["sync_to_sub_distance_histogram"], bins, ax, "green"
    )
    assert count_sum1 == count_sum3

    count_sum = count_sum1

    step = count_sum // 6
    step = int(round(step, int(-math.log10(step))))

    plt.yticks(list(range(0, count_sum - 1 - step, step)) + [count_sum], fontsize=10)
    # ax.axvline(x=0.2,color='black',linewidth=0.8,linestyle='dotted')
    ax.axhline(y=count_sum, color="black", linewidth=0.8, linestyle="dotted", xmin=0.2)

    ax.legend(
        [
            "Without synchronization",
            "With synchronization to video",
            "With synchronization to subtitle",
        ]
    )
    ax.set(
        xlabel="Time distance of lines between reference subtitle and incorrect subtitles",
        ylabel="Frequency (cumulative sum)",
    )
    ax.set_xscale("log")
    ax.set_xlim((startrange, endrange))
    ax.set_ylim((0, count_sum * 1.1))

    def format_func(value, tick_number):
        if value < 1:
            return "{}ms".format(int(value * 1000))
        else:
            return "{}s".format(int(value))

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    # plt.yscale('log')
    # plt.xscale('log')

    plt.savefig(os.path.join(output_dir, "distance-histogram." + extension))

    # plt.show()
    plt.close()
