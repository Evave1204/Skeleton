import skeletor as sk
import trimesh as tm
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial import Delaunay
from skeletor import Skeleton

def show1(lines, points):
    path = tm.path.Path3D(entities=lines, vertices=points)
    scene = tm.Scene()
    scene.add_geometry(path)
    scene.show()

class Node:
  def __init__(self, id=None, coords=[]):
    self.id = id
    self.coords= coords
class Branch:
  def __init__(self, branch_num=None, end_node=None, start_node=None, interval_node=[], edge_raddi=None, edge_angle=0):
    self.branch_num = branch_num
    self.end_node = end_node
    self.start_node = start_node
    self.edge_raddi = edge_raddi
    self.edge_angle = edge_angle
    self.interval_node = interval_node
def get_branch(branch_num, start_node_id, end_node_id, interval_nodes):
  start_node = Node(id=start_node_id, coords=skel.vertices[start_node_id])
  end_node = Node(id=end_node_id, coords=skel.vertices[end_node_id])
  branch = Branch(branch_num=branch_num,start_node=start_node, end_node=end_node, interval_node=interval_nodes)
  return branch
def get_tree(edge_map, current_node_id, start_node_id, res, 
               branch_num, Branch, Node, interval = False, interval_nodes=[]):
  # print("-----------------")
  # print("Branch Num:", branch_num)
  # print("current node:", current_node_id)
  # print("start_node:", start_node_id)
  # print([node.id for node in interval_nodes])
  if current_node_id in edge_map:
    if interval and len(edge_map[current_node_id]) == 1:
      # in the interval
      node = Node(id=current_node_id, coords=skel.vertices[current_node_id])
      interval_nodes.append(node)
    if len(edge_map[current_node_id]) == 1:
      # go to next node
      current_node_id = edge_map[current_node_id][0]
      res = get_tree(edge_map, current_node_id, start_node_id, res,
                            branch_num, Branch, Node, interval=True, interval_nodes=interval_nodes)
    else:
      # add the whole branch
      branch_num += 1
      end_node_id = current_node_id
      #radii = radius[start_node_id]
      br = get_branch(branch_num, start_node_id, end_node_id, interval_nodes)
      if branch_num not in res: res[branch_num] = [br]
      else: res[branch_num].append(br)
      start_node_id = current_node_id  
      for node in edge_map[current_node_id]:
        current_node_id = node
        res = get_tree(edge_map, current_node_id, start_node_id, res,
                            branch_num, Branch, Node, interval=False, interval_nodes=[])
    return res
  else:
    branch_num += 1
    end_node_id = current_node_id
    br = get_branch(branch_num, start_node_id, end_node_id, interval_nodes)
    if branch_num not in res: res[branch_num] = [br]
    else: res[branch_num].append(br)
    return res
def reverse_tree(tree, new_root):
    # 构建原始树的逆向映射
    parent_map = {}
    for parent, children in tree.items():
        for child in children:
            if child in parent_map:
                parent_map[child].append(parent)
            else:
                parent_map[child] = [parent]

    # 使用新根节点构建新的树
    new_tree = {new_root: []} 
    print("tree:", tree)
    print("parent_map:",parent_map)
    print("new_tree:",new_tree)

    def add_child(tree, child_nodes, new_tree):
        for child_node in child_nodes:
            if child_node in tree:
                new_tree[child_node] = tree[child_node]
                child_nodes = tree[child_node]
                new_tree = add_child(tree, child_nodes, new_tree)     
        return new_tree
            
    def build_tree(node, new_tree, tree, pre_node=None):
        # tree : {1: [2], 2: [3, 4], 3: [5]}
        # parent_map : {2: [1], 3: [2], 4: [2], 5: [3]}
        # new_tree : {5: [3], 3: [2], 2: [1, 4]}
        # node : 5
        if node in parent_map:
          #print(new_tree, parent_map[node])
          for parent_node in parent_map[node]:
              # parent_node -> child_node BECOME child_node -> parent_node
              if node in new_tree:
                  new_tree[node].append(parent_node)
              else:
                  new_tree[node] = [parent_node]
              
              # add original child node to new tree and child node's children are also added
              if node in tree:
                  for child_node in tree[node]:
                      if child_node != pre_node:
                        new_tree[node].append(child_node)
                        new_tree = add_child(tree, [child_node], new_tree)

              # recursive
              if parent_node in tree:
                  pre_node = node
                  new_tree = build_tree(parent_node, new_tree, tree, pre_node)
        return new_tree
    
    new_tree = build_tree(new_root, new_tree,tree)
    return new_tree
def get_relation(res):
  branch_num = 1
  res_map = {}
  while branch_num in res:
    brs = res[branch_num]
    for br in brs:
      res_map[br.start_node.id] = br.branch_num
      for node in br.interval_node:
        res_map[node.id] = br.branch_num
      res_map[br.end_node.id] = br.branch_num if br.end_node else None
    print(res_map)
    branch_num += 1
  return res_map

class TreeNode:
    def __init__(self, value, coords, parent=None):
        self.value = value
        self.level = None
        self.coords = coords
        self.children = []
        self.parent = parent
        self.edges = {}

    def add_child(self, child_node, edge_radii=None, edge_length=None, edge_angle=None):
        self.children.append(child_node)
        self.edges[child_node] = {'radii': edge_radii, 'length':edge_length, 'angle':edge_angle}

from collections import deque, defaultdict       
def assign_levels(root, branch_num={}):
    # 使用队列进行广度优先搜索，记录每个节点的深度
    queue = deque([(root, 0)])
    node_depth = defaultdict(int)
    node_depth[0] = [root]
    leaf_nodes = []

    while queue:
        node, depth = queue.popleft()
        is_leaf = True
        for child in node.children:
            queue.append((child, depth + 1))
            if depth + 1 not in node_depth:
                node_depth[depth + 1] = [child]
            else:
                node_depth[depth + 1].append(child)
            is_leaf = False
        if is_leaf:
            leaf_nodes.append(node)
    # 初始化所有叶子节点的层级为1
    leafs = []
    branch_res = {"Level1":{"Number":0, "Angle":0, "Length":0, "Radii":0}}
    for leaf in leaf_nodes:
        leafs.append(leaf.value)
        leaf.level = 1
        branch_res["Level1"]["Number"] += 1

    
    # 从叶子节点开始逐层向上更新每个节点的层级
    down_depth = list(node_depth)
    down_depth.sort(reverse=True)
    for d in down_depth:
        nodes = node_depth[d]
        for node in nodes:
            t_num = False
            if node.children:
                child_levels = [child.level for child in node.children]
                #print(child_levels)
                if len(child_levels) == 1:
                    node.level = child_levels[0]
                elif len(child_levels) == len(list(set(child_levels))):
                    node.level = max(child_levels)
                else:
                    dic = {}
                    for level in child_levels:
                        if level not in dic:
                            dic[level] = 1
                        else:
                            dic[level] += 1
                    levels = list(dic.keys())
                    levels.sort(reverse=True)
                    new_levels = []
                    for level in levels:
                        if dic[level] > 1:
                            new_levels.append(level)
                    max_new = max(new_levels)
                    max_old = max(levels)
                    #print(new_levels, levels)
                    if max_new >= max_old:
                        # [3,3,1] 
                        node.level = max_new + 1
                        t_num = True
                    else:
                        # [3,1,1]
                        node.level = max_old
                if t_num:
                    #print(branch_res)
                    branch_res = assign_number(branch_res, node.level)
    return branch_res

def assign_number(branch_res, level):
    flag = "Level" + str(level)
    if flag not in branch_res:
        branch_res[flag] = {"Number":1, "Angle":0, "Length":0, "Radii":0}#, "Angle":angle}
    else:
        branch_res[flag]["Number"] += 1
    return branch_res


# 输出节点的层数
def print_tree_levels(node, dic, lines, colors, goals, k=None, high_level = 0, branch_res={}):
    for child in node.children:
        parent_point = node.value
        child_point = child.value
        if child.level < node.level and node.edges[child]['angle']:
            angle = node.edges[child]['angle']
            branch_res["Level"+str(child.level)]["Angle"] += angle
        if node.edges[child]['length']:
            length = node.edges[child]['length']
            branch_res["Level"+str(child.level)]["Length"] += length
        if node.edges[child]['radii']:
            radii = node.edges[child]['radii']
            branch_res["Level"+str(child.level)]["Radii"] += radii
        points = [parent_point, child_point]
        line = tm.path.entities.Line(points=points)
        if child.level < len(colors) and (not k or (parent_point in k or child_point in k)):
            line.color = colors[child.level - 1]
        else:
            line.color = [0,0,0,255]
        if child.level == high_level:
            goals.append([node, node.edges[child]])
        lines.append(line)
        if child.level not in dic:
            dic[child.level] = [{"Radii":node.edges[child]['radii'], "Length":node.edges[child]['length']}]
        else:
            dic[child.level].append({"Radii":node.edges[child]['radii'], "Length":node.edges[child]['length']})
        #print(f"{node.value}->{edge['to'].value} branch_level: {edge['to'].level} radius: {edge['radii']} length: {edge['length']} angle: {edge['angle']}")
    #print("---------------------------------------------------------------------------------------------------------------------------------------")
    for child in node.children:
        print_tree_levels(child, dic,lines, colors, goals, k=k, high_level=high_level, branch_res=branch_res)
    return dic, lines, goals, branch_res

def contruct_tree(tree, current_node, skel):
    root = TreeNode(current_node, skel.vertices[current_node])
    def build_tree(node, root, tree):
        if node in tree:
            for child in tree[node]:
                id = child[0]
                radii = child[1]
                #if node < len(skel.vertices) and id < len(skel.vertices):
                length = np.linalg.norm(skel.vertices[node] - skel.vertices[child[0]])
                child_node = TreeNode(id, skel.vertices[id], parent=root)
                if root.parent == None:
                    root.add_child(child_node, edge_radii=radii, edge_length=length)
                else:
                    angle_radians, angle_degrees = calculate_angle_between_vectors(child_node.coords, root.coords, root.parent.coords) 
                    root.add_child(child_node, edge_radii=radii, edge_length=length, edge_angle=angle_degrees)
                build_tree(id, child_node, tree)
    build_tree(current_node, root, tree)
    return root

import math
def calculate_angle_between_vectors(A1, A2, A3):
    # 计算向量 A1A2 和 A2A3
    vector_A1A2 = (A2[0] - A1[0], A2[1] - A1[1], A2[2] - A1[2])
    vector_A2A3 = (A3[0] - A2[0], A3[1] - A2[1], A3[2] - A2[2])
    
    # 计算向量的点积
    dot_product = (vector_A1A2[0] * vector_A2A3[0] +
                   vector_A1A2[1] * vector_A2A3[1] +
                   vector_A1A2[2] * vector_A2A3[2])
    
    # 计算向量的模长
    magnitude_A1A2 = math.sqrt(vector_A1A2[0]**2 + vector_A1A2[1]**2 + vector_A1A2[2]**2)
    magnitude_A2A3 = math.sqrt(vector_A2A3[0]**2 + vector_A2A3[1]**2 + vector_A2A3[2]**2)
    
    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude_A1A2 * magnitude_A2A3)
    
    # 计算夹角（弧度）
    angle_radians = math.acos(cos_theta)
    
    # 转换夹角为度数
    angle_degrees = math.degrees(angle_radians)
    
    return angle_radians, angle_degrees

def reverse_tree(tree, new_root):
    # 构建原始树的逆向映射
    del tree[-1]
    parent_map = {}
    for parent, children in tree.items():
        for child in children:
            #if child in parent_map:
            if child[0] in parent_map:
                #parent_map[child].append(parent)
                parent_map[child[0]].append([parent,child[1]])
            else:
                #parent_map[child] = [parent]
                parent_map[child[0]] = [[parent, child[1]]]

    # 使用新根节点构建新的树
    new_tree = {new_root: []} 
    print("tree:", tree)
    print("parent_map:",parent_map)
    print("new_tree:",new_tree)

    def add_child(tree, child_nodes, new_tree):
        for child_node in child_nodes:
            # if child_node in tree:
            #     new_tree[child_node] = tree[child_node]
            #     child_nodes = tree[child_node]
            #     new_tree = add_child(tree, child_nodes, new_tree)
            if child_node[0] in tree:
                new_tree[child_node[0]] = tree[child_node[0]]
                child_nodes = tree[child_node[0]]
                new_tree = add_child(tree, child_nodes, new_tree)     
        return new_tree
            
    def build_tree(node, new_tree, tree, pre_node=None):
        # tree : {1: [2], 2: [3, 4], 3: [5]}
        # parent_map : {2: [1], 3: [2], 4: [2], 5: [3]}
        # new_tree : {5: [3], 3: [2], 2: [1, 4]}
        # node : 5
        
        if node in parent_map:
          #print(new_tree, parent_map[node])
          for parent_node in parent_map[node]:
              parent_id = parent_node[0]
              parent_radii = parent_node[1]
              # parent_node -> child_node BECOME child_node -> parent_node
              if node in new_tree:
                  new_tree[node].append([parent_id, parent_radii])
              else:
                  new_tree[node] = [[parent_id, parent_radii]]
              
              # add original child node to new tree and child node's children are also added
              if node in tree:
                  for child_node in tree[node]:
                      if child_node[0] != pre_node:
                      #if child_node != pre_node:
                        #new_tree[node].append(child_node)
                        new_tree[node].append([child_node[0],child_node[1]])
                        new_tree = add_child(tree, [child_node], new_tree)

              # recursive
              if parent_id in tree:
                  pre_node = node
                  new_tree = build_tree(parent_id, new_tree, tree, pre_node)
        return new_tree
    
    new_tree = build_tree(new_root, new_tree,tree)
    return new_tree
# 修改 show 方法以接受骨架颜色和背景颜色参数
# 绑定新方法到 skel 对象
def scene(self, mesh=False, **kwargs):
    """Return a Scene object containing the skeleton.

    Returns
    -------
    scene :     trimesh.scene.scene.Scene
                Contains the skeleton and optionally the mesh.

    """
    for enty in self.skeleton.entities:
        if isinstance(enty, tm.path.entities.Line):
            enty.color = [255, 0, 0, 255]
    if mesh:
        if isinstance(self.mesh, type(None)):
            raise ValueError('Skeleton has no mesh.')

        self.mesh.visual.face_colors = [100, 100, 100, 100]

        # Note the copy(): without it the transform in show() changes
        # the original meshes
        sc = tm.Scene([self.mesh.copy(), self.skeleton.copy()], **kwargs)
    else:
        sc = tm.Scene(self.skeleton.copy(), **kwargs)

    return sc

def show(self, mesh=False, **kwargs):
    """Render the skeleton in an opengl window. Requires pyglet.

    Parameters
    ----------
    mesh :      bool
                If True, will render transparent mesh on top of the
                skeleton.

    Returns
    --------
    scene :     trimesh.scene.Scene
                Scene with skeleton in it.

    """
    scene = self.scene(mesh=mesh)

    # I encountered some issues if object space is big and the easiest
    # way to work around this is to apply a transform such that the
    # coordinates have -5 to +5 bounds
    fac = 5 / np.fabs(self.skeleton.bounds).max()
    scene.apply_transform(np.diag([fac, fac, fac, 1]))

    return scene.show(**kwargs)
Skeleton.scene = scene
Skeleton.show = show