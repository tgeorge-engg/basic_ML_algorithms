import numpy as np

#add pruning, add visualizer,
class DecisionTreeClassifier:
    def __init__(self):
        self.title=""
        self.branches=[]
        self.level=0
        self.split_ind=None
        self.is_cat=True
        self.num_split_val=None
        self.classification=None
        self.cats=None
        self.print_str=None

    #decision tree visualization
    def __str__(self):
        if self.print_str is None:
            self.__unify(self.branches)
            tree_list=self.__traverse()
            initialized=self.__init_fold_tree(tree_list)
            tree_str=self.__fold_tree(initialized)
            self.print_str=tree_str
        return self.print_str

    def __traverse(self):
        if self.branches == []:
            return self.title
        else:
            branch_list = []
            for branch in self.branches:
                branch_list.append(branch.__traverse())
            return [self.title, branch_list]

    def __unify(self, branch_list):
        c = 0
        for branch in branch_list:
            if branch.branches != []:
                c = 1
                break

        if c == 0:
            return

        continue_list = []

        for branch in branch_list:
            if not branch.branches:
                branch.branches = [DecisionTreeClassifier(), DecisionTreeClassifier()]
                branch.branches[0].title="    "
                branch.branches[1].title="    "
            for branch2 in branch.branches:
                continue_list.append(branch2)

        return self.__unify(continue_list)

    def __init_leaf(self, root, branch_list):
        root_len = len(root)
        branch_num = len(branch_list)

        root_spacer = sum([len(branch) for branch in branch_list[:branch_num // 2]]) + (
                    (branch_num - 1) // 2) * root_len

        root_str = " " * root_spacer + root
        branch_str = (" " * (root_len // 2 + 1)).join(branch_list)

        return root_str + "\n" + branch_str

    def __init_fold_tree(self, tree_list):
        if type(tree_list[1][0]) == str:
            return self.__init_leaf(tree_list[0], tree_list[1])
        else:
            for i in range(len(tree_list[1])):
                tree_list[1][i] = self.__init_fold_tree(tree_list[1][i])
            return tree_list

    def __combine_node(self, root, branches):
        root_len = len(root)
        branch_num = len(branches)
        branch_strs = [branch.split('\n') for branch in branches]
        branch_spacers = []

        # child_spacer[i]=spacers in order of level; first level 0, then level 1, ...
        for branch in branch_strs:
            branch_i_spacer = []
            for i in range(len(branch) - 1):
                branch_i_spacer.append(len(branch[-1]) - len(branch[i]))
            branch_i_spacer.append(0)
            branch_spacers.append(branch_i_spacer)

        # adding the spaces to the child_string
        for i in range(1, len(branch_strs)):
            for j in range(len(branch_strs[i])):
                branch_strs[i][j] = " " * branch_spacers[i - 1][j] + " " * root_len + branch_strs[i][j]

        # spacing for parent
        if len(branch_strs) % 2 == 1:
            root_spacer = sum([len(branch[-1]) for branch in branch_strs[:(branch_num // 2) + 1]]) - \
                            branch_spacers[branch_num // 2][0] - len(branch_strs[branch_num // 2][0].strip(' '))
        else:
            root_spacer = sum([len(branch[-1]) for branch in branch_strs[:branch_num // 2]]) + root_len * (
                        (branch_num // 2) - 1)

        root_str = " " * root_spacer + root
        combined_branch_strings = "\n".join(
            ["".join([branch_strs[j][i] for j in range(len(branch_strs))]) for i in range(len(branch_strs[0]))])

        return root_str + "\n" + combined_branch_strings

    def __fold_tree(self, mod_tree_list):
        # if condition for final string needed?
        if type(mod_tree_list[1][0]) == str:
            return self.__combine_node(mod_tree_list[0], mod_tree_list[1])
        else:
            for i in range(len(mod_tree_list[1])):
                mod_tree_list[1][i] = self.__fold_tree(mod_tree_list[1][i])
            return self.__fold_tree(mod_tree_list)
    ##end visualization

    #data must be in order of training data
    def classify(self, pred_data):
        if not self.branches:
            return self.classification
        else:
            if self.is_cat:
                return self.branches[self.cats.index(pred_data[self.split_ind])].classify(pred_data)
            else:
                if pred_data[self.split_ind]<self.num_split_val:
                    return self.branches[0].classify(pred_data)
                else:
                    return self.branches[1].classify(pred_data)


    def train(self, data, result, split_limit=0.85, num_reuse_limit=0.4, min_sample_size=2, max_height=None):
        self.__tree_make(data, result, split_limit, num_reuse_limit, min_sample_size)
        if type(max_height)==int:
            self.__prune(max_height)
        return

    #data=list of lists
    def __tree_make(self, data, result, split_limit, num_reuse_limit, min_sample_size):

        min_gini=1.1
        split_ind=0
        split_cats=None
        split_crit=None

        #check if data has to be split more
        uniform, uniform_category = self.__check_uniform(result, split_limit)
        skip1=True
        for i in data:
            if i:
                skip1=False

        if uniform or len(result)<=min_sample_size or skip1:
            self.title=str(uniform_category)
            self.classification=uniform_category
            return

        #find split category and split value
        for i in range(len(data)):
            if data[i]:
                if self.__is_categorical(data[i]):
                    gini_i,split_cat_i=self.__gini_cat(data[i], result)
                    split_crit_i = None

                else:
                    gini_i,split_crit_i=self.__gini_num(data[i], result)
                    split_cat_i = None

                if gini_i<min_gini:
                    min_gini=gini_i
                    split_ind=i
                    split_cats=split_cat_i
                    split_crit=split_crit_i

        #recursive split
        ##checking if numerical category should be split again
        self.split_ind=split_ind
        if split_crit is not None:
            self.title = f"<{split_crit}"
            self.num_split_val=split_crit
            self.is_cat=False
            data_new = [[[] for j in range(len(data) + 1)] for i in range(2)]
            for i in range(len(data[0])):
                if data[split_ind][i]<split_crit:
                    move_ind=0
                else:
                    move_ind=1
                for j in range(len(data)):
                    if data[j]:
                        data_new[move_ind][j].append(data[j][i])
                data_new[move_ind][-1].append(result[i])

            if num_reuse_limit>min_gini:
                for cat_data in data_new:
                    # cat_data.pop(split_ind)
                    cat_data[split_ind] = []

            self.branches = [DecisionTreeClassifier() for i in range(2)]
            for i in range(len(data_new)):
                self.branches[i].__tree_make(data_new[i][:-1], data_new[i][-1], split_limit, num_reuse_limit, min_sample_size)
        else:
            self.title = " ".join(split_cats)
            self.cats=split_cats
            cat_inds={split_cats[i]:i for i in range(len(split_cats))}
            data_new=[[[] for j in range(len(data)+1)] for i in range(len(split_cats))]
            for i in range(len(result)):
                move_ind=cat_inds[data[split_ind][i]]
                for j in range(len(data)):
                    if data[j]:
                        data_new[move_ind][j].append(data[j][i])
                data_new[move_ind][-1].append(result[i])
            for cat_data in data_new:
                cat_data[split_ind]=[]
            self.branches=[DecisionTreeClassifier() for i in range(len(split_cats))]
            for i in range(len(data_new)):
                self.branches[i].__tree_make(data_new[i][:-1],data_new[i][-1], split_limit, num_reuse_limit, min_sample_size)
        return

    def __prune(self, max_height,current_height=0):
        self.level=current_height
        if current_height>=max_height:
            self.branches=[]
            return
        for branch in self.branches:
            branch.__prune(max_height, current_height+1)
        return

    def __check_uniform(self, result, split_limit):
        vals = {}
        for result_i in result:
            if result_i in vals:
                vals[result_i] += 1
            else:
                vals[result_i] = 1
        proportion = max(vals.values()) / sum(vals.values())
        if proportion >= split_limit:
            return True, max(vals, key=vals.get)
        else:
            return False, max(vals, key=vals.get)

    def __is_categorical(self, data_i, unique_limit=5):
        if type(data_i[0]) == str or type(data_i[0]) == bool:
            return True
        uniques = []
        for point in data_i:
            if point not in uniques:
                uniques.append(point)
            if len(uniques) > unique_limit:
                return False
        return True

    def __gini_cat(self, data_i, result):
        data_cats, placeholder = np.unique(data_i, return_counts=True)
        result_cats, placeholder = np.unique(result, return_counts=True)
        data_inf = {d_cat: {r_cat: 0 for r_cat in result_cats} for d_cat in data_cats}

        for i in range(len(data_i)):
            data_inf[data_i[i]][result[i]] += 1

        sub_ginis = [1 - sum([dp ** 2 for dp in data_inf[cat].values()]) / (sum(data_inf[cat].values()) ** 2) for cat in
                     data_cats]
        weighted_ginis = [sub_ginis[i] * sum(data_inf[data_cats[i]].values()) / len(data_i) for i in
                          range(len(data_cats))]
        return sum(weighted_ginis),data_cats.tolist()

    def __gini_num(self, data_i, result, step=2):
        sort_dr = sorted(zip(data_i, result), key=lambda pair: pair[0])
        sort_data = [i for i, _ in sort_dr]
        sort_result = [i for _, i in sort_dr]
        result_cats, placeholder = np.unique(result, return_counts=True)
        data_cats = ["l", "r"]
        data_inf = {"l": {r_cat: 0 for r_cat in result_cats}, "r": {r_cat: 0 for r_cat in result_cats}}

        min_gini = 1.1
        split_val = np.mean([sort_data[i] for i in range(step)])

        for i in range(len(sort_result)):
            data_inf["r"][result[i]] += 1

        for i in range(0, len(sort_data) - step + 1, step - 1):
            for j in range(i, i + step - 1):
                data_inf["l"][result[j]] += 1
                data_inf["r"][result[j]] -= 1
            sub_ginis = [1 - sum([dp ** 2 for dp in data_inf[cat].values()]) / (sum(data_inf[cat].values()) ** 2) for
                         cat in
                         data_cats]
            weighted_ginis = [sub_ginis[i] * sum(data_inf[data_cats[i]].values()) / len(data_i) for i in
                              range(len(data_cats))]

            tot_gini = sum(weighted_ginis)

            if tot_gini < min_gini:
                min_gini = tot_gini
                split_val = np.mean([sort_data[j] for j in range(i, i + step)])

        return min_gini, split_val

#simple example
data_s1=['a1','a1','a1','a1','a2','a2','a2','a2']
data_s2=['a','b','a','b','a','a','b','b',]
data_s=[data_s1, data_s2]
result_s=[0,0,0,0,1,1,0,0]
dt1=DecisionTreeClassifier()
dt1.train(data_s,result_s,min_sample_size=0)
print(dt1)
print("-"*30)

#more complex example
data_c1=['a','a','a','a','a','a','b','b','b','b','c','c','c','c','c','c']
data_c2=['a1','a1','b1','b1','c1','c1','a1','c1','b1','b1','a1','a1','b1','b1','c1','c1']
data_c3=['a11','b11','a11','b11','a11','b11','b11','a11','a11','b11','a11','b11','a11','b11','a11','b11']
data_c=[data_c1, data_c2, data_c3]
result_c=["cat","cat","dog","dog","cat","dog","cat","cat","cat","dog","dog","dog","dog","dog","dog","dog"]
dt2=DecisionTreeClassifier()
dt2.train(data_c,result_c,min_sample_size=0)
print(dt2)
print("-"*30)

#decision tree != optimal tree
data_n1=['a','a','a','a','b','b','b','b']
data_n2=[1,3,7,9,2,4,20,42]
data_n=[data_n1,data_n2]
result_n=[0,0,1,1,0,0,1,1]
dt3=DecisionTreeClassifier()
dt3.train(data_n,result_n,min_sample_size=0)
print(dt3)