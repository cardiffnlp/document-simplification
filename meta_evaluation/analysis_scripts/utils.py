from scipy.stats import pearsonr
            
def pairwise_kendall(dataset, metric_name):
    agg_ratings_dimension = {}
    for instance in dataset:
        for rating in instance["ratings"]:
            name = rating["name"]
            mvals = instance["metrics"][metric_name]
            agg_ratings_dimension.setdefault(name, {
                "concordant": 0, "discordant": 0
            })

            pairwise_mval = 0
            mval_diff = abs(mvals["simplification2"] - mvals["simplification1"])

            # We remove comparisions with small difference in metric values.
            if mvals["simplification2"] > mvals["simplification1"]:
                pairwise_mval = 1
            human = rating["agg_value"]
                
            if pairwise_mval == human:
                agg_ratings_dimension[name]["concordant"] += 1
            else:
                agg_ratings_dimension[name]["discordant"] += 1

    taus = []
    for dimen, counts in agg_ratings_dimension.items():
        concordant = counts["concordant"]
        discordant = counts["discordant"]
        if (concordant + discordant) > 0:
            tau = (concordant - discordant) / (concordant + discordant)
            taus.append(tau)
    return sum(taus) / len(taus)



def pointwise_pearson(dataset, metric_name, dimensions=None):
    all_corrs = []
    if dimensions is None:
        dimensions = [rating["name"] for rating in dataset[0]["ratings"]]
    print(dimensions)
    for dimension in dimensions:
        mvals, human_vals = [], []
        for instance in dataset:
            rating = [ratings["agg_value"] for ratings in instance['ratings'] 
                      if ratings['name'] == dimension][0]
            human_vals.append(rating)
            mvals.append(instance["metrics"][metric_name])
        assert len(mvals) == len(human_vals)
        corr, _ = pearsonr(mvals, human_vals)
        all_corrs.append(corr)
    return sum(all_corrs) / len(all_corrs)
