"""Patient-local multivariate intensity/local-mean/local-SD habitats."""
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import regional_habitat_features_20261012 as shared

PROTOCOL = {**shared.PROTOCOL,
    "partition":"patient-local K=3 KMeans on intensity, 3-mm local mean and local SD; refit per contour; order by intensity center",
    "quantiles":None,"partition_sampling":"up to 4000 current-mask voxels, seed64212; n_init5",
    "partition_normalization":"current-patient mask-relative values and descriptor scaler; never pooled patients or outcomes",
    "spatial_fraction":"variable habitat occupancy included when training robustness and sign-stability gates pass"}


def partition(values,mask):
    mask = np.asarray(mask,bool);labels = np.zeros(mask.shape,np.uint8)
    if int(mask.sum())<12:return labels,np.array([np.nan,np.nan])
    v = np.asarray(values,float);inside = v[mask]
    mean,sd = float(inside.mean()),float(inside.std())
    if sd<=1e-12:
        labels[mask]=1
        return labels,np.array([mean,mean])
    z = np.where(mask,(v-mean)/sd,0.)
    support = uniform_filter(mask.astype(float),size=3,mode="constant")
    localmean = uniform_filter(z,size=3,mode="constant")/np.maximum(support,1e-12)
    second = uniform_filter(z*z,size=3,mode="constant")/np.maximum(support,1e-12)
    localsd = np.sqrt(np.maximum(second-localmean**2,0.))
    x = np.column_stack([z[mask],localmean[mask],localsd[mask]])
    rng = np.random.default_rng(64212)
    ix = rng.choice(len(x),min(4000,len(x)),replace=False)
    scaler = StandardScaler().fit(x[ix]);normalized = scaler.transform(x)
    model = KMeans(n_clusters=3,n_init=5,random_state=64212).fit(normalized[ix])
    order = np.argsort(model.cluster_centers_[:,0],kind="stable")
    remap = np.empty(3,np.uint8);remap[order]=np.arange(1,4)
    labels[mask]=remap[model.predict(normalized)]
    physical_centers = scaler.inverse_transform(model.cluster_centers_)[:,0]*sd+mean
    return labels,np.array([physical_centers[order[0]],physical_centers[order[-1]]])


def extract(profiles,indices,ids):
    original = shared.partition
    shared.partition = partition
    try:data = shared.extract(profiles,indices,ids)
    finally:shared.partition = original
    data["protocol"] = PROTOCOL
    for row in data["partition_audit"]:
        row["LowClusterMean"] = row.pop("CutLow")
        row["HighClusterMean"] = row.pop("CutHigh")
    return data
