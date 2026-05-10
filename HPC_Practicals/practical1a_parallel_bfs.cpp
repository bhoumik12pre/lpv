#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

const int MAXN = 100005;
vector<int> adj[MAXN];
bool visited[MAXN];

void parallel_bfs(int start, int n) {
    fill(visited, visited + n + 1, false);

    vector<int> current = {start};
    visited[start] = true;

    while (!current.empty()) {
        vector<int> next;

        #pragma omp parallel
        {
            vector<int> local;

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < (int)current.size(); i++) {
                int node = current[i];

                for (int nb : adj[node]) {
                    bool already;

                    #pragma omp critical
                    {
                        already = visited[nb];

                        if (!already)
                            visited[nb] = true;
                    }

                    if (!already)
                        local.push_back(nb);
                }
            }

            #pragma omp critical
            next.insert(next.end(), local.begin(), local.end());
        }

        cout << "Level: ";
        for (int v : current)
            cout << v << " ";

        cout << endl;

        current = next;
    }
}

int main() {
    int n = 6;

    int edges[][2] = {
        {0,1},{0,2},{1,3},
        {1,4},{2,4},{3,5},{4,5}
    };

    for (auto &ed : edges) {
        adj[ed[0]].push_back(ed[1]);
        adj[ed[1]].push_back(ed[0]);
    }

    cout << "Parallel BFS from node 0:" << endl;

    parallel_bfs(0, n);

    return 0;
}