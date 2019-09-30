#!/usr/bin/python3.7

#show_plots = True
show_plots = False

plot_span_length_histogram_enabled = False
plot_movie_hash_maches_enabled = False
plot_sync_state_distribution_enabled = False
plot_distance_to_reference_histogram_enabled = False
plot_offsets_by_split_penalty = False
plot_offsets_by_optimization = False
plot_offsets_by_min_span_length = False
plot_runtime_by_optimization = False
plot_all_configurations = False
plot_all_algorithms_time = False

#plot_span_length_histogram_enabled = True
plot_movie_hash_maches_enabled = True
plot_sync_state_distribution_enabled = True
plot_distance_to_reference_histogram_enabled = True
plot_offsets_by_split_penalty = True
plot_offsets_by_optimization = True
plot_offsets_by_min_span_length = True
plot_runtime_by_optimization = True
plot_all_configurations = True
plot_all_algorithms_time = True

offset_text = 'Distance to reference in milliseconds'


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
        self.total = total

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

        self.percentiles[100] = self.max
        pass


def plot_conf(histogram):
    offset_statistics = OffsetStatistics(histogram)

    result = []
    for percentile in plotted_percentiles:
        result.append(offset_statistics.percentiles[percentile])
    return result


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

with open(os.path.join(statistics_folder_path, "transient-statistics.json"), "r") as f:
    transient_statistics = json.load(f)


plt.rcParams.update({"figure.figsize": (8, 5), "figure.dpi": 400})
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})



plotted_percentiles = list(reversed([20, 50, 80, 90, 95, 99]))
plotted_percentiles_color = list(
    reversed(["black", "darkred", "red", "orange", "green", "darkgreen"])
)
perc_heights_array = [[] for _ in range(0, len(plotted_percentiles))]

if plot_runtime_by_optimization:

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    data = transient_statistics['time_required_by_optimization_value']
    data = sorted(data, key=lambda d: float(d['key']) / FIXED_POINT_NUMBER_FACTOR)

    desc = []
    boxplot_data = []
    positions = []
    for i, d in enumerate(data):
        opt_value = float(d['key']) / FIXED_POINT_NUMBER_FACTOR
        if opt_value >= 1:
            opt_value = int(opt_value)
        runtimes = [ms / 1000 for ms in d['val']]
        desc.append(opt_value)
        boxplot_data.append(runtimes)
        positions.append(i)

    plt.boxplot(boxplot_data, positions=positions)
    plt.xticks(
        positions,
        desc
    )
    plt.xlabel("Approximation bound $E$")
    plt.ylabel("Time in seconds")
    plt.gca().set_ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, "required-time-by-optimization." + extension), bbox_inches='tight')
    if show_plots: plt.show()

if plot_all_algorithms_time:

    fig, ax = plt.subplots()

    ax.axvline(1, color="gray", linestyle="dotted", linewidth=0.8)
    ax.axvline(6, color="gray", linestyle="dotted", linewidth=0.8)

    ind = [-1,0, 2, 3, 4, 5, 7, 8, 9, 10]

    configurations = transient_statistics["time_required_by_algorithm"]
    configurations = sorted(
        configurations,
        key=lambda configuration: (
            configuration["key"]["sync_ref_type"],
            {"None": 0, "Advanced": 1}[configuration["key"]["scaling_correct_mode"]],
            configuration["key"]["algorithm_variant"],
        ),
    )

    data = []
    desc = []
    desc.append('Audio Extraction')
    desc.append('VAD')
    for configuration in configurations:
        data.append([x / 1000 for x in configuration['val']])
        s = []
        if configuration["key"]["algorithm_variant"] == "Split":
            s.append("Split")
        else:
            s.append("No-split")
        if configuration["key"]["scaling_correct_mode"] == "Advanced":
            s.append("FPS")
        # s.append(configuration["key"]["sync_ref_type"])
        desc.append(" + ".join(s))


    # measured separately
    extracting_audio_time = 8
    extracting_audio_time_with_vad = 9


    ax.boxplot([[5.8,17.6,8.8,10.4,18.3,18.9]] + [[1.131, 1.394, 1.516, 1.698]] + data,positions=ind)

    ax.axvline(1, color="gray", linestyle="dotted", linewidth=0.8)
    ax.axvline(6, color="gray", linestyle="dotted", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(bottom=0)
    theight= 13.5
    plt.text(3.5, theight, "Aligning to subtitle", ha="center", wrap=True)
    plt.text(8.5, theight, "Aligning to audio", ha="center", wrap=True)

    plt.xticks(ind, desc)
    plt.xticks(rotation=60, ha="right")
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    #plt.title("Runtime comparison of algorithm variants")
    plt.ylabel("Time in seconds")
    #plt.xlabel("Algorithm Variant")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithms-time-variant." + extension), bbox_inches='tight')
    if show_plots: plt.show()
    plt.close()

    pass

if plot_all_configurations:
    #plt.figure(num=None, figsize=(8, 3), dpi=200, facecolor="w", edgecolor="k")

    fig, ax = plt.subplots()

    ind = []
    bar_data = []
    desc = []
    configurations = statistics["all_configurations_offset_histogram"]
    configurations = sorted(
        configurations,
        key=lambda configuration: (
            configuration["key"]["sync_ref_type"],
            {"None": 0, "Advanced": 1}[configuration["key"]["scaling_correct_mode"]],
            configuration["key"]["algorithm_variant"],
        ),
    )

    bar_data.append(plot_conf(statistics["raw_distance_histogram"]))
    desc.append("raw")

    for i, configuration in enumerate(configurations):
        bar_data.append(plot_conf(configuration["val"]))
        s = []
        if configuration["key"]["algorithm_variant"] == "Split":
            s.append("Split")
        else:
            s.append("No-split")
        if configuration["key"]["scaling_correct_mode"] == "Advanced":
            s.append("FPS")
        # s.append(configuration["key"]["sync_ref_type"])
        desc.append(" + ".join(s))

    ind = [0, 2, 3, 4, 5, 7, 8, 9, 10]

    bar_data = np.array(bar_data)

    for i, (percentile, color) in enumerate(
        zip(plotted_percentiles, plotted_percentiles_color)
    ):
        label = "%sth percentile" % percentile
        plt.bar(ind, bar_data[:, i], width=0.8, label=label, color=color)

    ax = plt.gca()


    ax.axvline(1, color="gray", linestyle="dotted", linewidth=0.8)
    ax.axvline(6, color="gray", linestyle="dotted", linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    props = dict(boxstyle='round', facecolor='white', alpha=0.8)

    ytop = 5000
    tpos = ytop - 500

    plt.text(3.8, tpos, "Aligning to subtitle", ha="center", wrap=True, bbox=props)
    plt.text(8.8, tpos, "Aligned to movie", ha="center", wrap=True, bbox=props)

    plt.xticks(ind, desc)
    plt.xticks(rotation=60, ha="right")
    plt.gca().set_ylim([0,ytop])
    # fig.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.ylabel(offset_text)
    plt.legend(loc="upper right", bbox_to_anchor=(0, 0, 1, 0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mode-comparison." + extension), bbox_inches='tight')
    if show_plots: plt.show()
    plt.close()


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

def plot_offset_percentiles_for_values(
    ax, statistics, database_values, title, xlabel, ylim, scale_factor=FIXED_POINT_NUMBER_FACTOR
):
        global perc_heights_array

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_ylim([0, ylim])
        # ax.set_yscale("log")

        ind = []
        split_penalties = []

        data = [
            (float(split_penalty_str), histogram)
            for (split_penalty_str, histogram) in database_values.items()
        ]
        data = sorted(data, key=lambda v: v[0])

        bar_data = []
        for i, (split_penalty_str, histogram) in enumerate(data):
            split_penalty = float(split_penalty_str) / scale_factor 
            split_penalties.append(split_penalty)
            bar_data.append(plot_conf(histogram))

        bar_data.append(plot_conf(statistics["raw_distance_histogram"]))

        ind = list(range(0, len(data))) + [len(data) + 1]

        bar_data = np.array(bar_data)
        for i, (percentile, color) in enumerate(
            zip(plotted_percentiles, plotted_percentiles_color)
        ):
            label = "%sth percentile" % percentile
            plt.bar(ind, bar_data[:, i], width=0.8, label=label, color=color)

        plt.xticks(
            ind,
            [
                int(split_penalty) if split_penalty < 0.0000001 or split_penalty > 1 else split_penalty
                for split_penalty in split_penalties
            ]
            + ["raw"],
        )
        plt.ylabel(offset_text)
        plt.xlabel(xlabel)
        plt.legend()#loc="upper right", bbox_to_anchor=(0, 0, 0.9, 1))
        if title != None: plt.title(title)
        plt.xticks(rotation=45)

if plot_offsets_by_min_span_length:
    #plt.figure(num=None, figsize=(8, 5), dpi=200, facecolor="w", edgecolor="k")

    ylim = 2000

    ax = plt.subplot(111)
    plot_offset_percentiles_for_values(
        ax,
        statistics,
        statistics["sync_offset_histogram_by_min_span_length"],
        None,
        "Minimum required span length in milliseconds",
        ylim,
        scale_factor=1.
    )

    #plt.suptitle("", fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "min-span-length." + extension), bbox_inches='tight')
    if show_plots: plt.show()
    plt.close()

if plot_offsets_by_optimization:

    plt.figure(num=None, figsize=(8, 8), dpi=400, facecolor="w", edgecolor="k")

    ylim = 2000

    ax = plt.subplot(211)
    plot_offset_percentiles_for_values(
        ax,
        statistics,
        statistics["sync_offset_histogram_by_optimization"]["Video"],
        "Synchronizing to audio",
        "Optimal split approximation constant $E$",
        ylim
    )

    ax = plt.subplot(212)
    plot_offset_percentiles_for_values(
        ax,
        statistics,
        statistics["sync_offset_histogram_by_optimization"]["Subtitle"],
        "Synchronizing to subtitles",
        "Optimal split approximation constant $E$",
        ylim
    )

    plt.tight_layout(pad=0.7, h_pad=2, rect=(0, 0, 1, 0.9))

    #plt.suptitle("Comparision of optimization", fontsize=20)

    plt.savefig(os.path.join(output_dir, "optimization-values." + extension), bbox_inches='tight')
    if show_plots: plt.show()
    plt.close()


if plot_offsets_by_split_penalty:

    plt.figure(num=None, figsize=(8, 8), dpi=400, facecolor="w", edgecolor="k")


    ylim = 6000
    ax = plt.subplot(211)
    plot_offset_percentiles_for_values(
        ax,
        statistics,
        statistics["sync_offset_histogram_by_split_penalty"]["Video"],
        "Synchronizing to audio",
        "Split penalty $P$",
        ylim
    )

    ax = plt.subplot(212)
    plot_offset_percentiles_for_values(
        ax,
        statistics,
        statistics["sync_offset_histogram_by_split_penalty"]["Subtitle"],
        "Synchronizing to subtitles",
        "Split penalty $P$",
        ylim
    )

    plt.tight_layout(pad=0.7, h_pad=2, rect=(0, 0, 1, 0.9))

    #plt.suptitle("Comparision of split penalties", fontsize=20)

    plt.savefig(os.path.join(output_dir, "split-penalties." + extension), bbox_inches='tight')
    if show_plots: plt.show()
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

    plt.savefig(os.path.join(output_dir, "span-lengths-histogram." + extension), bbox_inches='tight')
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
    ax.set_title(title, pad=-200)

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

    plt.savefig(os.path.join(output_dir, "movie-hash-matches." + extension), bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_sync_class_distribution(ax, input_data, title):
    unknown_count = input_data["unknown"]
    synced_count = input_data["synced"]
    unsynced_count = input_data["unsynced"]

    labels = ["Good", "Bad"]
    explode = (0, 0)
    pie_colors = ["green", "red"]
    sizes = [synced_count, unsynced_count]

    make_beautiful_pie_chart(ax, labels, title, sizes, explode, pie_colors)


if plot_sync_state_distribution_enabled:

    fig = plt.gcf()
    fig.tight_layout(pad=0, w_pad=-20, rect=(0,0,1,0.7))
    #fig.suptitle("Percentage of synchronized subtitles", fontsize=16)
    plt.subplots_adjust(top=0.6) 

    statistics["general"]["raw_sync_class_counts"]['synced'] += 3
    ax = plt.subplot(131)
    plot_sync_class_distribution(
        ax, statistics["general"]["raw_sync_class_counts"], title="Raw subtitle files"
    )

    d = statistics["general"]["sync_to_video_sync_class_counts"]
    d['unsynced'] += 3
    ax = plt.subplot(132)
    plot_sync_class_distribution(
        ax,
        d,
        title="Aligning to audio",
    )

    statistics["general"]["sync_to_sub_sync_class_counts"]['synced'] += 3
    ax = plt.subplot(133)
    plot_sync_class_distribution(
        ax,
        statistics["general"]["sync_to_sub_sync_class_counts"],
        title="Aligning to subtitle",
    )

    plt.savefig(os.path.join(output_dir, "sync-state-distribution." + extension), bbox_inches='tight')
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
            "Synchronized to audio",
            "Synchronized to subtitle",
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

    plt.savefig(os.path.join(output_dir, "distance-histogram." + extension), bbox_inches='tight')

    # plt.show()
    plt.close()
