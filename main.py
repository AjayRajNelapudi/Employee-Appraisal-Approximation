import csv
import numpy as np
from sklearn.cluster import KMeans

MIN_APPRAISAL = 25
MAX_APPRAISAL = 75
NO_OF_GROUPS = 10

def read_datafile(dataset_file):
    with open(dataset_file) as dataset_file:
        dataset_reader = csv.reader(dataset_file)
        dataset = list(dataset_reader)[1:]
        dataset = np.array([list(map(float, datapoint))
                            for datapoint in dataset])
        return dataset

def train(dataset):
    model = KMeans(n_clusters=NO_OF_GROUPS)
    model.fit(dataset)
    return model

def weight(center):
    weights = np.array([0.05/4, 0.2/4, 0.2/4, 0.1/1, 0.3/4, 0.05/4, 0.05/40, 0.1/40, 0.05/18])
    datapoint_weight = sum([feature * weight for feature, weight in zip(center, weights)])
    return datapoint_weight

def sort_by_appraisal(cluster_centers_):
    cluster_centers = [list(center) for center in cluster_centers_]
    cluster_centers.sort(key=weight)
    return cluster_centers

def appraise(model, datafile, sorted_centers):
    samples = read_datafile(datafile)
    labels = model.predict(samples)

    appraisals = []
    for sample, label in zip(samples, labels):
        center = list(model.cluster_centers_[label])
        center_index = sorted_centers.index(center)
        appraisal_percentage = (center_index + 1) * ((MAX_APPRAISAL - MIN_APPRAISAL) / NO_OF_GROUPS) + MIN_APPRAISAL
        appraisals.append({'datapoint': sample, 'appraisal': appraisal_percentage})

    return appraisals

def main():
    dataset = read_datafile("dataset.csv")
    model = train(dataset)
    sorted_centers = sort_by_appraisal(model.cluster_centers_)

    appraisals = appraise(model, "samples.csv", sorted_centers)

    for appraisal in appraisals:
        print(appraisal)

if __name__ == "__main__":
    main()
