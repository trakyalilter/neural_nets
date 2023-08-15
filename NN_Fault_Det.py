import os
import pandas as pd
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dropout, Concatenate, concatenate
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Conv2D, MaxPooling2D


#SAVE DATAS IN NPZ FORMAT
#np.savez_compressed()
exp1 = pd.read_csv("wax_machining_ds/experiment_01.csv")
res = pd.read_csv("wax_machining_ds/train.csv")
print(exp1.head(5))

Machining_Process_Values = exp1["Machining_Process"].unique()

exp1.replace(Machining_Process_Values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)


# Loop through the groups and create separate heatmaps
def plot_and_save(exp1,num_groups, flag_=True):
    if flag_ == True:
        num_groups = 16
        features_per_group = len(exp1.columns) // num_groups
        for group in range(num_groups):
            start_idx = group * features_per_group
            end_idx = (group + 1) * features_per_group if group != num_groups - 1 else None

            # Subset the DataFrame to the current group of features
            subset_df = exp1.iloc[:, start_idx:end_idx]

            # Calculate correlations
            correlations = subset_df.corr()

            # Create a heatmap
            plt.figure(figsize=(10, 8))
            sbn.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title(f"Correlation Heatmap - Group {group + 1}")
            if not os.path.exists("Correlation_Plots_Vax_Machining"):
                os.makedirs("Correlation_Plots_Vax_Machining")
            plt.savefig(f"Correlation_Plots_Vax_Machining/Correlation Heatmap-Group {group + 1}")
            #plt.show()


plot_and_save(exp1,16, False)

plt.scatter(np.arange(exp1["X1_ActualPosition"].shape[0]), (exp1["X1_ActualPosition"] - exp1["X1_CommandPosition"]))
#plt.show()

plt.scatter(np.arange(exp1["X1_ActualVelocity"].shape[0]), (exp1["X1_ActualVelocity"] - exp1["X1_CommandVelocity"]))
#plt.show()

filtered_data = res[
    (res["passed_visual_inspection"].isna())
]
res.fillna("no", inplace=True)



res.replace(res["tool_condition"].unique(), [1, 0], inplace=True)
res.replace(res["machining_finalized"].unique(), [1, 0], inplace=True)
res.replace(res["passed_visual_inspection"].unique(), [1, 0], inplace=True)
print(res)

res["clamp_pressure"] = res["clamp_pressure"] / res["clamp_pressure"].abs().max()
res["feedrate"] = res["feedrate"] / res["feedrate"].abs().max()
print(res)

frames = []
lengths = []

_ = pd.read_csv("wax_machining_ds/experiment_01.csv")

_.replace(_["Machining_Process"].unique(), np.arange(1, len(_["Machining_Process"].unique()) + 1), inplace=True)

paths_list = []
for i in range(1, 19):
    if i < 10:
        paths_list.append(f"wax_machining_ds/experiment_0{i}.csv")
    else:
        paths_list.append(f"wax_machining_ds/experiment_{i}.csv")

for i in range(1, 18 + 1):
    if i < 10:
        _ = pd.read_csv(f"wax_machining_ds/experiment_0{i}.csv").to_numpy()
    else:
        _ = pd.read_csv(f"wax_machining_ds/experiment_{i}.csv").to_numpy()

    np.savez_compressed(fr"C:\IT\PythonProject\OffsetCorrection\wax_machining_ds\as_np\num_{i}.npz", _)
    _ = np.load(fr"C:\IT\PythonProject\OffsetCorrection\wax_machining_ds\as_np\num_{i}.npz")
    frames.append(_)
    lengths.append(len(_))
merged_df = pd.DataFrame({"Experiment Data": frames})

plt.title("Lengths of Data Distribution")
plt.scatter(np.arange(1, 19), lengths)
max_rows = lengths[np.argmax(lengths)]
_ = pd.read_csv("wax_machining_ds/experiment_01.csv")

_.replace(_["Machining_Process"].unique(), np.arange(1, len(_["Machining_Process"].unique()) + 1), inplace=True)
for i in range(0, max_rows - _.shape[0]):
    _.loc[len(_.index)] = np.zeros(_.shape[1])

plt.xlim([0, 18])
plt.grid()
#plt.show()
padded_frames = []

padded_lengths = []
command_columns = ['X1_CommandPosition', 'X1_CommandVelocity', 'X1_CommandAcceleration',
                   'Y1_CommandPosition', 'Y1_CommandVelocity', 'Y1_CommandAcceleration',
                   'Z1_CommandPosition', 'Z1_CommandVelocity', 'Z1_CommandAcceleration',
                   'S1_CommandPosition', 'S1_CommandVelocity', 'S1_CommandAcceleration']

actual_columns = ['X1_ActualPosition', 'X1_ActualVelocity', 'X1_ActualAcceleration',
                  'Y1_ActualPosition', 'Y1_ActualVelocity', 'Y1_ActualAcceleration',
                  'Z1_ActualPosition', 'Z1_ActualVelocity', 'Z1_ActualAcceleration',
                  'S1_ActualPosition', 'S1_ActualVelocity', 'S1_ActualAcceleration']

def padding_experiments_data():
    d=0
    for path,k in zip(paths_list,np.arange(1,len(paths_list)+1)):
        if not os.path.exists(f"{path[:-4]}-padded.csv"):
            _ = pd.read_csv(path)
            _.replace(_["Machining_Process"].unique(), np.arange(1, len(_["Machining_Process"].unique()) + 1),
                      inplace=True)
            for i in range(0, max_rows - _.shape[0]):
                _.loc[len(_.index)] = np.zeros(_.shape[1])
            _.to_csv(f"{path[:-4]}-padded.csv")

        else:
            _ = pd.read_csv(f"{path[:-4]}-padded.csv")

        _.rename(columns={'Unnamed: 0': 'Steps'}, inplace=True)
        padded_lengths.append(_.shape[0])
        for cmd_col, actual_col in zip(command_columns, actual_columns):
            _[f'{cmd_col}_Offset'] = _[cmd_col] - _[actual_col]

        # Drop the original Command and Actual columns
        _.drop(columns=command_columns + actual_columns, inplace=True)
        __ = _
        _ = _.to_numpy()
        num_features = __.shape[1]
        for i in range(num_features):
            plt.figure(figsize=(8,6))
            plt.plot(np.arange(1,__.shape[0]+1,),__.iloc[:,i])
            plt.xlabel('Index')
            plt.ylabel(f'{__.columns[i]} Values')
            plt.title(f' Experiment {k} \n Plot of {__.columns[i]}')
            plt.grid()
            if not os.path.exists(fr"C:\IT\PythonProject\OffsetCorrection\wax_machining_ds\plots\experiment_{k}"):
                os.makedirs(fr"C:\IT\PythonProject\OffsetCorrection\wax_machining_ds\plots\experiment_{k}")
                plt.savefig(fr"C:\IT\PythonProject\OffsetCorrection\wax_machining_ds\plots\experiment_{k}\plot_of_{__.columns[i]}")

            plt.close('all') #close or plt.figure.close()
            #plt.show()
        #plt.tight_layout()
        #plt.show()
        if d == 0:
            plot_and_save(__, 16, True)

            d += 1

        padded_frames.append(_)

    return padded_frames

#plot_and_save(exp1,16, False)
padding_experiments_data()
X = np.stack(padded_frames)
"""res["Experiment Data"] = pd.DataFrame({"Experiment Data": padded_frames})"""

y = res["passed_visual_inspection"].values
res.drop(columns=["passed_visual_inspection"], inplace=True)
X_variables = res.iloc[:, :4].values
"""X = res.iloc[:, -1]
X = res.iloc[:, -1].values"""

from sklearn.model_selection import train_test_split

X_train, X_test, X_variables_train, X_variables_test, y_train, y_test = train_test_split(
    X, X_variables, y, test_size=0.4, random_state=42)
print(X.shape)
shape_Cnn = (2332,37)
cnn_model = Input(shape=shape_Cnn)
x = Flatten()(cnn_model)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)


mlp_model = Input(shape=(4,))
y = Dense(64, activation='relu')(mlp_model)


combined = concatenate([x, y])

z = Dense(16, activation='relu')(combined)
output = Dense(1, activation='sigmoid')(z)

print("Types:",type(X_train),type(X_variables_train),type(y_train),sep="\n")
print("Whole:",X_train)
print("First:",X_train[0])
print("2nd:",X_train[0][0])
model = Model(inputs=[cnn_model, mlp_model], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("X VALUES TO FIT\n*************************************************\n",[X_train, X_variables_train])
print(model.summary())


history = model.fit([X_train, X_variables_train], y_train, epochs=100, batch_size=32, validation_split=0.2)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.close('all')
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Loss")
plt.show()

plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("Accuracy")
plt.show()
print("Training Ended")

test_loss, test_accuracy = model.evaluate([X_test, X_variables_test],
                                          y_test)