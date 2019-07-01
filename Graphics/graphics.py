import numpy as np
import matplotlib 
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.rcParams["savefig.jpeg_quality"] = 100
matplotlib.rcParams["savefig.dpi"] = 2000
from matplotlib import path, rcParams
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime


class Results_writter():
    """
    Class Results_writter:
    - Parameters:
        model_name: Vector containing the image paths
        config: Parameters to reproduce experiment
        results: obtained_results
        predicts: model predictionx
        labels: original labels

    """
    def __init__(self, model_name, config, results, predictions, labels, data_actual):
        self.model_name = model_name
        self.config = config
        self.results = results
        self.cm = confusion_matrix(labels, predictions)
        self.class_names = self.translate_idx_class_name()
        print(self.class_names)
        self.data_actual = data_actual


    def translate_idx_class_name(self):
        idx_2_name = np.load('../Data_management/class2folder.npy').item()
        class_2_snake_csv = np.loadtxt(open("class_id_maapping.csv", "rb"), dtype = str, delimiter=",", skiprows=1)
        class_2_snake = {}
        for line in class_2_snake_csv:
            class_2_snake[line[0]]= line[1]
        class_names = []
        for k,v in idx_2_name.items():
            print(v,class_2_snake.values())
            class_names.append(list(class_2_snake.keys())[list(class_2_snake.values()).index(str(v.split('-')[-1]))])
        return class_names


    def save(self):
        # Conf matrix
        self.plot_confusion_matrix(self.cm,'confusion_'+self.model_name+'_'+self.data_actual, self.class_names  )

        # Losses and accus
        self.plot_losses(self.results, 'losses_'+self.model_name+'_'+self.data_actual)

        # Save vectors of losses and accus
        self.save_results_disk('res_'+self.model_name+'_'+self.data_actual)

        #self.insert_sql()

    def save_results_disk(self, filename):
        np.save(filename, self.results)



    def insert_sql(self):
        '''
        Inserta nom del model, vector losses, vector accuracies
        '''
        insert = """INSERT INTO public.training_data (model_name, id_exp, 
            train_loss, val_loss, train_acc, val_acc, best_accuracy_val) \
            VALUES(%s, %s,%s, %s, %s, %s, %s );"""

        try:
            # create a new cursor
            cur = self.conn.cursor()
            # execute the INSERT statement

            cur.execute(insert, (self.model_name, self.data_actual, 
                list(self.results['losses']['train']), list(self.results['losses']['val']),
                list(self.results['acc']['train']), list(self.results['acc']['val']), max(list(self.results['acc']['val'])) ))
            # commit the changes to the database
            self.conn.commit()
            # close communication with the database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        return

    def plot_confusion_matrix(self, matrix  ,filename , labels , title='Confusion matrix'):
        fig, ax  = plt.subplots()
        ax.set_xticks([x for x in range(len(labels))])
        ax.set_yticks([y for y in range(len(labels))])
        # Place labels on minor ticks
        ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
        ax.set_xticklabels(labels, rotation='90', fontsize=5, minor=True)
        ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
        ax.set_yticklabels(labels[::-1], fontsize=5, minor=True)
        # Hide major tick labels
        ax.tick_params(which='major', labelbottom='off', labelleft='off')
        # Finally, hide minor tick marks
        ax.tick_params(which='minor', width=0)

        # Plot heat map
        proportions = [1. * row / sum(row) for row in matrix]
        ax.pcolor(np.array(proportions[::-1]), cmap=plt.cm.Blues)

        # Plot counts as text
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                confusion = matrix[::-1][row][col]
                if confusion != 0:
                    ax.text(col + 0.5, row + 0.5, confusion, fontsize=2,
                        horizontalalignment='center',
                        verticalalignment='center')

        # Add finishing touches
        ax.grid(True, linestyle=':')
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(filename+'.png')
        #cleaning memory
        plt.cla()
        plt.clf()
        plt.close(fig)
        return filename+'.png'

    def plot_losses(self, data , filename ):

        plt.subplot(2,1,1)

        plt.plot(data['losses']['train'], c = 'b', label = 'train')
        plt.plot(data['losses']['val'], c = 'r', label = 'validation')
        plt.title('Loss: Categorical Cross-Entropy')
        plt.legend(loc='upper right')
        plt.subplot(2,1,2)
        plt.plot(data['acc']['train'], c = 'b', label = 'train')
        plt.plot(data['acc']['val'], c = 'r', label = 'validation')
        plt.legend(loc='upper right')
        plt.title('Accuracy')
        plt.savefig(filename+'.png')
        #cleaning memory
        plt.cla()
        plt.clf()
        plt.close()
        return filename+'.png'
    



if __name__ == '__main__':
    data_actual = '2019-06-27_18-57-51'

    path = '/home/jordi/Desktop/Snake_results/'
    res = np.load(path+'results_'+data_actual+'.npy').item()
    pred = np.load(path+'predictions_'+data_actual+'.npy')
    labels = np.load(path+'val_labels_'+data_actual+'.npy')
    grafer = Results_writter('resnet101', '', res, pred, labels, data_actual)
    grafer.save()