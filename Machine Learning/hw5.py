import numpy
from classifiers import create_decision_tree, create_random_forest, calculate_model_accuracy, calculate_confusion_matrix
from data import get_minecraft, get_first_n_samples

def p0(featuretype='histogram'):
    data_train, data_test, target_train, target_test = get_minecraft(featuretype)
    model = create_decision_tree()

    # TODO: Fit the model to the data using its fit method
    model.fit(data_train, target_train)

    # TODO: Use the model's predict method to predict labels for the training and test sets
    predict_train = model.predict(data_train[0:])
    predict_test = model.predict(data_test[0:])

    accuracy_train, accuracy_test = calculate_model_accuracy(predict_train, predict_test, target_train, target_test)
    print('Training accuracy: {0:3f}, Test accuracy: {1:3f}'.format(accuracy_train, accuracy_test))

    cfm = calculate_confusion_matrix(predict_test,target_test)
    print "Confusion matrix"
    print cfm

    for q in range(1,3):
        for p in range(0,q):
            #compute confusion between classes p and q
            index_pq = [i for i,v in enumerate(target_train) if v in [p,q]]
            modelpq = create_decision_tree()
            #TODO: fit model to the data only involving classes p and q
            testindex_pq = [i for i,v in enumerate(target_test) if v in [p,q]]

            p_target_train = []
            p_data_train = []

            for elem in index_pq:
                p_target_train.append(target_train[elem])
                p_data_train.append(data_train[elem])

            p_data_test = []
            p_target_test = []
            for elem in testindex_pq:
                p_target_test.append(target_test[elem])
                p_data_test.append(data_test[elem])

            modelpq.fit(p_data_train, p_target_train)

            predict_TRAIN = modelpq.predict(p_data_train[0:])
            predict_TEST = modelpq.predict(p_data_test[0:])

            #TODO: calculate and print the accuracy
            accuracy_train, accuracy_test = calculate_model_accuracy(predict_TRAIN, predict_TEST, p_target_train, p_target_test)
            accuracy_pq = accuracy_test
            print "One-vs-one accuracy between classes",p,"and",q,":",accuracy_pq

    return model, predict_train, predict_test, accuracy_train, accuracy_test


def p1():
    #TODO: compare different feature types
    #m,ptrain,ptest,atrain,atest = p0('histogram')
    #m,ptrain,ptest,atrain,atest = p0('rgb')
    m,ptrain,ptest,atrain,atest = p0('gray')

def p2():
    results = []
    model = None

    # TODO: Get the Minecraft dataset using get_minecraft() and create a decision tree
    data_train, data_test, target_train, target_test = get_minecraft('histogram')

    for n in [50, 100, 150, 200, 250]:
        # TODO: Fit the model using a subset of the training data of size n
        # Hint: use the get_first_n_samples function imported from data.py
        data, target = get_first_n_samples(data_train, target_train, n)
        data_t, target_t = get_first_n_samples(data_test, target_test, n)

        model = create_decision_tree()

        model.fit(data, target)

        #TODO: use the model to fit the training data and predict labels for the training and test data
        predict_train = model.predict(data[0:])
        predict_test = model.predict(data_t[0:])

        # TODO: Calculate the accuracys of the model (use the training data that fit the model in the current iteration)
        accuracy_train_n, accuracy_test = calculate_model_accuracy(predict_train, predict_test, target, target_t)
        print('Training accuracy: {0:3f}, Test accuracy: {1:3f}'.format(accuracy_train_n, accuracy_test))

        results.append((n, accuracy_train_n, accuracy_test))

    print(results)
    return model, results


def p3():
    results = []
    model = None

    # TODO: Get the Minecraft dataset
    data_train, data_test, target_train, target_test = get_minecraft()

    for n_estimators in [2, 5, 10, 20, 30]:
        # TODO: create a random forest classifier with n_estimators estimators
        model = create_random_forest(n_estimators)

        #TODO: use the model to fit the training data and predict labels for the training and test data
        model.fit(data_train, target_train)

        predict_train = model.predict(data_train[0:])
        predict_test = model.predict(data_test[0:])

        # TODO: calculate the accuracies of the models and add them to the results
        accuracy_train, accuracy_test = calculate_model_accuracy(predict_train, predict_test, target_train, target_test)

        results.append((n_estimators, accuracy_train, accuracy_test))

    print(results)
    return model, results


def bonus():
    results = []
    model = None

    # OPTIONAL: Repeat p0 using a logistic regression classifier


    return model, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", type=str, choices=['p0', 'p1', 'p2', 'p3', 'bonus'], help="The problem to run")
    args = parser.parse_args()

    if args.problem == 'p0':
        p0()
    elif args.problem == 'p1':
        p1()
    elif args.problem == 'p2':
        p2()
    elif args.problem == 'p3':
        p3()
    elif args.problem == 'bonus':
        bonus()
