# Using example test set -- works
from mil.data.datasets import musk1
from mil.metrics import AUC, BinaryAccuracy
from mil.validators import KFold
from mil.trainer.trainer import Trainer
from mil.models import LogisticRegression
from mil.bag_representation.mapping import DiscriminativeMapping
from mil.preprocessing import StandarizerBagsList

(bags_train, y_train), (bags_test, y_test) = musk1.load()

trainer = Trainer()
metrics = [AUC, BinaryAccuracy]
model = LogisticRegression(solver='liblinear', C=1, class_weight='balanced')
pipeline = [('scale', StandarizerBagsList()), ('disc_mapping', DiscriminativeMapping(m=30))]
trainer.prepare(model, preprocess_pipeline=pipeline ,metrics=metrics)
valid = KFold(n_splits=2, shuffle=True)
history = trainer.fit(bags_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)

# Using WSI dataset -- too slow
def pickleOpen(file_name):
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return(loaded_list)

bags_subset = pickleOpen('data/sophie_ML/bags_subset.pkl')
labels_subset = pickleOpen('data/sophie_ML/labels_subset.pkl')
X_train, X_test, y_train, y_test = train_test_split(bags_subset, labels_subset, stratify=labels_subset)

history = trainer.fit(X_train, y_train, sample_weights='balanced', validation_strategy=valid, verbose=1)