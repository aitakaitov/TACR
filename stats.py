import copy
import statistics
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    ad_counts = []
    with open(ad_counts_file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            domain, count = line.strip().split(';')
            count = int(count)
            ad_counts.append((domain, count))

    art_counts = []
    with open(art_counts_file, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            domain, count = line.strip().split(';')
            count = int(count)
            art_counts.append((domain, count))

    return art_counts, ad_counts


def basic_stats(counts, data_name):
    # average, median
    print(f'Average of {sum([c for d, c in counts]) / float(len(counts))} docs per domain')
    print(f'Median of {statistics.median([c for d, c in counts])} docs per domain')

    # min, max
    print(f'Minimum of {min([c for d, c in counts])} docs per domain')
    print(f'Maximum of {max([c for d, c in counts])} docs per domain')

    # total
    print(f'Total of {sum([c for d, c in counts])} docs')

    # distribution
    plt.title(f'{data_name} distribution of counts')
    _ = plt.hist([c for d, c in counts], bins=25)
    plt.savefig(f'histogram_{data_name}.png')
    plt.cla()


def compare_stats(art_counts_orig, ad_counts_orig):
    art_counts = copy.copy(art_counts_orig)
    ad_counts = copy.copy(ad_counts_orig)

    art_domains = [d for d, c in art_counts]
    ad_domains = [d for d, c in ad_counts]

    # get the mismatched domains
    yes_art_no_ad = []
    no_art_yes_ad = []
    yes_art_yes_ad = []
    [yes_art_no_ad.append(art_d) if art_d not in ad_domains else 0 for art_d in art_domains]
    [no_art_yes_ad.append(ad_d) if ad_d not in art_domains else 0 for ad_d in ad_domains]
    [yes_art_yes_ad.append(d) if d in ad_domains else None for d in art_domains]

    # mismatches
    print(f"Domains that have articles but don't have ads: {len(yes_art_no_ad)} [{', '.join(yes_art_no_ad)}]")
    print(f"Domains that have ads but don't have articles: {len(no_art_yes_ad)} [{', '.join(no_art_yes_ad)}]")
    print(f"Domains that have articles and ads: {len(yes_art_yes_ad)}")

    # get te

    # for d in yes_art_no_ad:
    #     ad_counts.append((d, 0))
    #
    # for d in no_art_yes_ad:
    #     art_counts.append((d, 0))

    art_counts_bkup = copy.copy(art_counts)
    ad_counts_bkup = copy.copy(ad_counts)
    art_counts = []
    ad_counts = []

    [art_counts.append((d, c)) if d in yes_art_yes_ad else None for d, c in art_counts_bkup]
    [ad_counts.append((d, c)) if d in yes_art_yes_ad else None for d, c in ad_counts_bkup]

    art_counts.sort(key=lambda x: x[0])
    ad_counts.sort(key=lambda x: x[0])

    #print([d for d, c in art_counts])
    #print([d for d, c in ad_counts])

    # fraction of ads per domain
    fractions_of_ads = []
    for (ad_d, ad_c), (art_d, art_c) in zip(ad_counts, art_counts):
        fractions_of_ads.append(float(ad_c) / float(ad_c + art_c))

    #print(fractions_of_ads)

    with open('fractions_of_ads.csv', 'w+', encoding='utf-8') as f:
        f.write('domain;art count;ad count;ad perc\n')
        for (d, ad_c), (d2, art_c), frac in zip(ad_counts, art_counts, fractions_of_ads):
            f.write(f'{d};{art_c};{ad_c};{frac}\n')

    # average, median, min, max
    print(f'Average of {sum(fractions_of_ads) / len(fractions_of_ads)}% of ads per domain')
    print(f'Median of {statistics.median(fractions_of_ads)}% of ads per domain')
    print(f'Minimum of {min(fractions_of_ads)}% of ads per domain')
    print(f'Maximum of {max(fractions_of_ads)}% of ads per domain')


def main():
    art_counts, ad_counts = load_data()

    print('--- ART BASIC STATS ---')
    basic_stats(art_counts, 'articles')

    print()
    print('--- AD BASIC STATS ---')
    basic_stats(ad_counts, 'ads')

    print()
    print('--- RATIOS ---')
    compare_stats(art_counts, ad_counts)



if __name__ == '__main__':
    ad_counts_file = 'ad_counts.csv'
    art_counts_file = 'art_counts.csv'
    main()