"""Grid A* for a disc robot among disc obstacles in a rectangular arena."""
import heapq
import math

import numpy as np

_N8 = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
       (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]


class Grid:
    """Occupancy grid. `obstacles` = [(x, y, r)] already inflated by the robot radius;
    `margin` = how far the robot centre must stay from the arena edge."""

    def __init__(self, arena, cell, margin, obstacles):
        self.cell, self.x0, self.y0 = cell, arena["xmin"], arena["ymin"]
        self.nx = int(round((arena["xmax"] - self.x0) / cell)) + 1
        self.ny = int(round((arena["ymax"] - self.y0) / cell)) + 1
        X, Y = np.meshgrid(self.x0 + np.arange(self.nx) * cell,
                           self.y0 + np.arange(self.ny) * cell, indexing="ij")
        e = 1e-9
        occ = ((X < arena["xmin"] + margin - e) | (X > arena["xmax"] - margin + e) |
               (Y < arena["ymin"] + margin - e) | (Y > arena["ymax"] - margin + e))
        for ox, oy, r in obstacles:
            occ |= (X - ox) ** 2 + (Y - oy) ** 2 < r * r
        self.occ = occ

    def cell_of(self, p):
        return (min(self.nx - 1, max(0, int(round((p[0] - self.x0) / self.cell)))),
                min(self.ny - 1, max(0, int(round((p[1] - self.y0) / self.cell)))))

    def point(self, ij):
        return (self.x0 + ij[0] * self.cell, self.y0 + ij[1] * self.cell)

    def free(self, p):
        return not self.occ[self.cell_of(p)]

    def nearest_free(self, p):
        fr = np.argwhere(~self.occ)
        if not len(fr):
            return None
        d = (self.x0 + fr[:, 0] * self.cell - p[0]) ** 2 + (self.y0 + fr[:, 1] * self.cell - p[1]) ** 2
        i, j = fr[int(np.argmin(d))]
        return int(i), int(j)

    def line_free(self, a, b):
        n = max(1, int(math.dist(a, b) / (self.cell * 0.5)))
        return all(self.free((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
                   for k in range(1, n + 1))


def plan(start, goal, obstacles, arena, cell=0.03, margin=0.05, info=None):
    """Waypoints from start to goal (start excluded), or None if there is no way.
    A start inside an obstacle first leaves by the nearest free cell; a blocked goal is
    replaced by the nearest free cell - `info["exact"]` then says False."""
    g = Grid(arena, cell, margin, obstacles)
    s, t, exact = g.cell_of(start), g.cell_of(goal), True
    if g.occ[s]:
        s = g.nearest_free(start)
    if g.occ[t]:
        t, exact = g.nearest_free(goal), False
    if info is not None:
        info["exact"] = exact
    if s is None or t is None:
        return None
    openq, came, cost = [(0.0, s)], {s: None}, {s: 0.0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == t:
            break
        for di, dj, w in _N8:
            n = (cur[0] + di, cur[1] + dj)
            if not (0 <= n[0] < g.nx and 0 <= n[1] < g.ny) or g.occ[n]:
                continue
            if di and dj and (g.occ[cur[0] + di, cur[1]] or g.occ[cur[0], cur[1] + dj]):
                continue                                   # no corner cutting
            c = cost[cur] + w
            if c < cost.get(n, 1e18):
                cost[n], came[n] = c, cur
                heapq.heappush(openq, (c + math.hypot(n[0] - t[0], n[1] - t[1]), n))
    if t not in came:
        return None
    cells, cur = [], t
    while cur is not None:
        cells.append(cur)
        cur = came[cur]
    pts = [g.point(c) for c in reversed(cells)]
    if exact:
        pts[-1] = (goal[0], goal[1])
    out, anchor, i = [], (start[0], start[1]), 0           # string-pull: skip waypoints in line of sight
    while i < len(pts):
        j = len(pts) - 1
        while j > i and not g.line_free(anchor, pts[j]):
            j -= 1
        out.append(pts[j])
        anchor, i = pts[j], j + 1
    return out
