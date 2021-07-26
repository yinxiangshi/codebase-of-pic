import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import pandas as pd
import os
from matplotlib.tri import Triangulation
import matplotlib

'''
this func could help you find out the trick range
And please customize the parameters u need:
alpha_range & beta_range
'''


def find_trick_range(file_path):
    files = os.listdir(file_path)
    iter = 0
    alpha_set = set()
    beta_set = set()
    for file in files:
        if not os.path.isdir(file):
            with open(file_path + '/' + file, 'r') as f:
                dict = json.load(f)
                f.close()
        alpha_range = dict['%s' % iter]['paras']['alpha_range']
        beta_range = dict['%s' % iter]['paras']['beta_range']
        alpha_set.add(alpha_range)
        beta_set.add(beta_range)
        iter+=1
        if len(alpha_set)>len(beta_set):
            m=len(alpha_set)
            n=len(beta_set)
        else:
            m=beta_set
            n=alpha_set
    return alpha_set, beta_set, m, n

'''
combine data in one cell
In default settings, beta_set is the y trick and alpha_set is the x trick.
'''


def process(m,n,file_path):
    alpha_set = set()
    beta_set = set()
    snorkel = np.zeros((m*n))
    mv = np.zeros((m*n))
    dp = np.zeros((m*n))
    fs = np.zeros((m*n))
    files = os.listdir(file_path)
    iter=0
    for file in files:
        if not os.path.isdir(file):
            with open(file_path + '/' + file, 'r') as f:
                dict = json.load(f)
                f.close()
        alpha = dict['%s' % iter]['paras']['alpha_range']
        beta = dict['%s' % iter]['paras']['beta_range']
        snorkel[iter] = dict['%s' % (iter+1)]['Snorkel']['result'][0]
        mv[iter] = dict['%s' % (iter+1)]['MajorityVoting']['result'][0]
        dp[iter] = dict['%s' % (iter+1)]['GeneralModel']['result'][0]
        fs[iter] = dict['%s' % (iter+1)]['FlyingSquid']['result'][0]
        alpha_set.add(alpha)
        beta_set.add(beta)
        iter += 1

    new_metai = np.zeros((n, m))
    new_dp = np.zeros((n, m))
    new_fs = np.zeros((n, m))
    new_mv = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            files = os.listdir(file_path)
            iter=0
            for file in files:
                with open(file_path+'/'+file, 'r') as f:
                    dict = json.load(f)
                    f.close()
                if dict['%s' % (iter+1)]['paras']['beta_range'] == beta_set[i] \
                        and dict['%s' % (iter+1)]['paras']['alpha_range'] == alpha_set[j]:
                    new_metai[n - i][j] = snorkel[iter]
                    new_mv[n - i - 1][j] = mv[iter]
                    new_fs[n - i - 1][j] = fs[iter]
                    new_dp[n - i - 1][j] = dp[iter]
                iter += 1
    values = [new_metai, new_mv, new_fs, new_dp]
    return values


def triangulation_for_triheatmap(M, N):
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]


if __name__ == '__main__':
    '''
    init parameters:
        file_path: string
    You could customize your pic's x trick and y trick by these parameters:        
        x_tricks: array
        y_tricks: array 
    notice: the y_trick need to be reverse order, x_trick is normal
    '''
    file_path = '/users/xxx/'
    x_tricks = []
    y_tricks = []
    '''
    Some init cong set:
    m: column
    n: row
    '''
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    m,n=1,1

    values = process(m,n,file_path)
    triangul = triangulation_for_triheatmap(m, n)
    cmaps = ['Blues', 'Greens', 'Purples', 'Reds']# define color u use
    norms = [plt.Normalize(-1, 0.5) for _ in range(4)]
    fig, ax = plt.subplots()
    imgs = [ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm, ec='white')
            for t, val, cmap, norm in zip(triangul, values, cmaps, norms)]
    imgs = [ax.tripcolor(t, val.ravel(), cmap='coolwarm', vmin=0.0, vmax=1.0, ec='white')
            for t, val in zip(triangul, values)]
    '''
    annotation
    '''
    for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
        for i in range(m):
            for j in range(n):
                v = val[j, i]
                ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}', color='k' if 0.2 < v < 0.8 else 'w',
                        ha='center', va='center', fontsize=17)

    cbar = fig.colorbar(imgs[0], ax=ax, fraction=0.031, pad=0.031)
    ax.set_xticks(range(m))
    ax.set_yticks(range(n))
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')

    font2 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 20,
             }

    cbar.ax.tick_params(labelsize=15)
    plt.xlabel('Diameter', font2)
    plt.ylabel('Max Acc.', font2)
    ax.set_title('10 labeling functions', fontsize=20, fontweight='bold')
    plt.tick_params(labelsize=15)
    plt.yticks(range(0, len(y_tricks)), y_tricks)
    plt.xticks(range(0, len(x_tricks)), x_tricks)

    fig.set_size_inches(12.5, 7.5)
    plt.tight_layout()
    plt.show()
    # n snorkel w dp e mv s fs


