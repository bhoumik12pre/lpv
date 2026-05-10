#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

const int MAXN = 100005;
vector<int> adj[MAXN];
bool visited[MAXN];

void parallel_dfs(int node) {
    visited[node] = true;

    cout << node << " ";

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)adj[node].size(); i++) {
        int nb = adj[node][i];

        bool already;

        #pragma omp critical
        {
            already = visited[nb];

            if (!already)
                visited[nb] = true;
        }

        if (!already)
            parallel_dfs(nb);
    }
}

int main() {
    int n = 6;

    fill(visited, visited + n + 1, false);

    int edges[][2] = {
        {0,1},{0,2},{1,3},
        {1,4},{2,4},{3,5},{4,5}
    };

    for (auto &ed : edges) {
        adj[ed[0]].push_back(ed[1]);
        adj[ed[1]].push_back(ed[0]);
    }

    cout << "Parallel DFS from node 0:" << endl;

    parallel_dfs(0);

    cout << endl;

    return 0;
}