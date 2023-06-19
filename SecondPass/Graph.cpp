/**
 * @file Graph.cpp
 * @author xiaomu
 * @brief 
 * @version 0.1
 * @date 2022-06-18
 * 
 * A quick second pass with the common 'Graph' algorithm problems.
 * 
 */

 /*
    463. Island Perimeter
    https://leetcode.com/problems/island-perimeter/
    You are given row x col grid representing a map where grid[i][j] = 1 represents land and 
    grid[i][j] = 0 represents water.

    Grid cells are connected horizontally/vertically (not diagonally). The grid is completely 
    surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

    The island doesn't have "lakes", meaning the water inside isn't connected to the water around 
    the island. One cell is a square with side length 1. The grid is rectangular, width and height 
    don't exceed 100. Determine the perimeter of the island.

    

    Example 1:
    Input: grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
    Output: 16
    Explanation: The perimeter is the 16 yellow stripes in the image above.

    Example 2:
    Input: grid = [[1]]
    Output: 4

    Example 3:
    Input: grid = [[1,0]]
    Output: 4
    

    Constraints:
    row == grid.length
    col == grid[i].length
    1 <= row, col <= 100
    grid[i][j] is 0 or 1.
    There is exactly one island in grid.
 */
// Solution 1: BFS
class Solution {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int res = 0;
        queue<pair<int, int>> Q;
        vector<vector<int>> visited(grid.size(), vector<int>(grid[0].size(), 0));
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[0].size(); ++ j) {
                if (grid[i][j] == 1) {
                    Q.push({i, j});
                    grid[i][j] = 2;
                    while (!Q.empty()) {
                        pair<int, int> island = Q.front();
                        int p = island.first, q = island.second;
                        Q.pop();
                        if (p - 1 < 0 || grid[p - 1][q] == 0) res ++;
                        if (p + 1 >= grid.size() || grid[p+1][q] == 0) res++;
                        if (q - 1 < 0 || grid[p][q-1] == 0) res ++;
                        if (q + 1 >= grid[0].size() || grid[p][q+1] == 0) res ++;

                        if (p - 1 >= 0 && grid[p-1][q] == 1) { 
                            Q.push({p-1, q});
                            // Needs to flip the value here to avoid duplicate visit.
                            grid[p-1][q] = 2;
                        }
                        if (p + 1 < grid.size() && grid[p+1][q] == 1) {
                            Q.push({p+1, q});
                            grid[p+1][q] = 2;
                        }
                        if (q - 1 >= 0 && grid[p][q-1] == 1) {
                            Q.push({p, q-1});
                            grid[p][q-1] = 2;
                        }
                        if (q + 1 < grid[0].size() && grid[p][q+1] == 1) {
                            Q.push({p, q+1});
                            grid[p][q+1] = 2;
                        }
                    }
                }
            }
        }
        return res;
    }
};

// Solution 2: get rid of the BFS. This problem does not require BFS.
class Solution {
public:
    int direction[4][2] = {0, 1, 1, 0, -1, 0, 0, -1};
    int islandPerimeter(vector<vector<int>>& grid) {
        int result = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 1) {
                    for (int k = 0; k < 4; k++) {       // 上下左右四个方向
                        int x = i + direction[k][0];
                        int y = j + direction[k][1];    // 计算周边坐标x,y
                        if (x < 0                       // i在边界上
                                || x >= grid.size()     // i在边界上
                                || y < 0                // j在边界上
                                || y >= grid[0].size()  // j在边界上
                                || grid[x][y] == 0) {   // x,y位置是水域
                            result++;
                        }
                    }
                }
            }
        }
        return result;
    }
};

// Count the maximum islands, then substract the repeated edges.
class Solution {
public://1035
    int islandPerimeter(vector<vector<int>>& grid) {
        if(grid.empty()) return 0;
        int count = 0, repeat = 0;
        for(int i = 0; i < grid.size(); i++)
            for(int j = 0; j < grid[0].size(); j++){
                if(grid[i][j] == 1){
                    count++;
                    // Only need to count two sides to avoid duplicate calculation.
                    if(i && grid[i - 1][j] == 1)  repeat++;
                    if(j && grid[i][j - 1] == 1)  repeat++;
                }
            }
        return 4 * count - 2 * repeat;
    }
};

 /*
    841. Keys and Rooms
    https://leetcode.com/problems/keys-and-rooms/
    There are n rooms labeled from 0 to n - 1 and all the rooms are locked except for room 0. 
    Your goal is to visit all the rooms. However, you cannot enter a locked room without having 
    its key.

    When you visit a room, you may find a set of distinct keys in it. Each key has a number on it, 
    denoting which room it unlocks, and you can take all of them with you to unlock the other rooms.

    Given an array rooms where rooms[i] is the set of keys that you can obtain if you visited 
    room i, return true if you can visit all the rooms, or false otherwise.

    

    Example 1:
    Input: rooms = [[1],[2],[3],[]]
    Output: true
    Explanation: 
    We visit room 0 and pick up key 1.
    We then visit room 1 and pick up key 2.
    We then visit room 2 and pick up key 3.
    We then visit room 3.
    Since we were able to visit every room, we return true.

    Example 2:
    Input: rooms = [[1,3],[3,0,1],[2],[0]]
    Output: false
    Explanation: We can not enter room number 2 since the only key that unlocks it is in that room.
    

    Constraints:

    n == rooms.length
    2 <= n <= 1000
    0 <= rooms[i].length <= 1000
    1 <= sum(rooms[i].length) <= 3000
    0 <= rooms[i][j] < n
    All the values of rooms[i] are unique.
 */
// Solution 1: BFS
class Solution {
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        int n = rooms.size();
        queue<int> Q;
        vector<int> visited(n, 0);
        Q.push(0);
        int count = 0;
        visited[0] = 1;
        while (!Q.empty()) {
            int room = Q.front();
            Q.pop();
            count++;
            if (count == n) return true;
            for (int i = 0; i < rooms[room].size(); ++i) {
                if (visited[rooms[room][i]] == 1) continue;
                Q.push(rooms[room][i]);
                visited[rooms[room][i]] = 1;
            }
        }
        return count == n;
    }
};

// DFS
class Solution {
private:
    int count = 0;
    bool dfs(vector<vector<int>>& rooms, vector<bool>& visited, int currentRoom) {
        if (visited[currentRoom]) return false;
        
        visited[currentRoom] = true;
        // We need to count all rooms that we can visit.
        count ++;
        if (count == rooms.size()) return true;
        for(int i = 0; i < rooms[currentRoom].size(); ++i) {
            if (dfs(rooms, visited, rooms[currentRoom][i])) return true;
        }
        return false;
    }
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        vector<bool> visited(rooms.size(), false);
        return dfs(rooms, visited, 0);
    }
};
