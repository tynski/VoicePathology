from sklearn.cluster import KMeans

def patientClassification(patientList):
	kmeans = KMeans(n_clusters = 2)
	kmeans.fit(patientList)

	return kmeans.labels_

