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