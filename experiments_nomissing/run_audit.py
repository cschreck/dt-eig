from experiments_nomissing.experiment_def import *
import pickle

dataset_name = 'audit'
X = pickle.load(open('../data/audit/X_audit.pkl', 'rb'))
y = pickle.load(open('../data/audit/y_audit.pkl', 'rb'))

if not os.path.exists('results'):
    os.makedirs('results')

np.seterr(all='ignore')
for i in range(5, 100, 5):
    print(i)
    missingness = i/100
    with open('./results/{}_results_{}.txt'.format(dataset_name, i), 'w') as f:
        res = experiment_setting_1(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")
        res = experiment_setting_2(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")
        res = experiment_setting_3(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")
        res = experiment_setting_4(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")
        res = experiment_setting_5(X, y, missingness=missingness)
        f.writelines(str(res)[1:-1] + "\n")

