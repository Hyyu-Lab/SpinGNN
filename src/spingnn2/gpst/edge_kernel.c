/* Fused GPST edge kernels: exact un-reduced and deterministic OpenMP shell-reducing.
 * Compile with: -std=c99 -O3 -fno-fast-math -ffp-contract=off -shared -fPIC [-fopenmp] -lm
 * No floating-point reassociation; OpenMP parallelizes strictly across independent frames t. */

#include <math.h>
#include <stdint.h>
#include <string.h>

static void gpst_poly(double x, int degree, double *p, double *d) {
    p[0] = 1.0;
    p[1] = x;
    d[0] = 0.0;
    d[1] = 1.0;
    for (int k = 1; k < degree; ++k) {
        p[k + 1] = ((2 * k + 1) * x * p[k] - k * p[k - 1]) / (k + 1);
        d[k + 1] = ((2 * k + 1) * (p[k] + x * d[k]) - k * d[k - 1]) / (k + 1);
    }
}

static double gpst_dot(const double *a, const double *b) {
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}

static void gpst_cross(const double *a, const double *b, double *c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

/* Exact un-reduced edge kernel (matches reference edge-array layout). */
int gpst_edges_exact(
    int64_t count, int64_t sites, int64_t edges,
    const double *s, const double *f, const int64_t *left, const int64_t *right,
    const double *w, const double *u, const double *zdir, const double *zforce,
    double *sn, double *sd,
    double *zn, double *zd, double *ph, double *cn, double *ss, double *an) {
    const int orders[4] = {1, 2, 4, 8};
    int bad = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:bad)
#endif
    for (int64_t t = 0; t < count; ++t) {
        for (int64_t e = 0; e < edges; ++e) {
            const double *a = s + (t * sites + left[e]) * 3;
            const double *b = s + (t * sites + right[e]) * 3;
            const double *fa = f + (t * sites + left[e]) * 3;
            const double *fb = f + (t * sites + right[e]) * 3;
            const double *v = u + e * 3;
            double x = gpst_dot(a, b), p[9], d[9], cparts[3];
            if (!isfinite(x) || fabs(x) > 1.0 + 1e-10) {
                bad = 1;
                continue;
            }
            x = fmin(1.0, fmax(-1.0, x));
            gpst_poly(x, 8, p, d);
            for (int k = 0; k < 3; ++k) cparts[k] = b[k] * fa[k] + a[k] * fb[k];
            double c = (cparts[0] + cparts[1]) + cparts[2];
            for (int k = 0; k < 4; ++k) {
                sn[(t * 4 + k) * edges + e] = (w[e] * d[orders[k]]) * c;
                sd[(t * 4 + k) * edges + e] = w[e] * p[orders[k]];
            }
            for (int k = 0; k < 2; ++k) {
                double z = zdir[(t * 2 + k) * edges + e], pp[9], dd[9];
                if (!isfinite(z) || fabs(z) > 1.0 + 1e-10) {
                    bad = 1;
                    continue;
                }
                z = fmin(1.0, fmax(-1.0, z));
                gpst_poly(z, 4, pp, dd);
                zn[(k * edges + e) * count + t] = (w[e] * dd[4]) * zforce[(t * 2 + k) * edges + e];
                zd[(k * edges + e) * count + t] = w[e] * pp[4];
            }
            double ab[3], bv[3], va[3];
            gpst_cross(a, b, ab);
            gpst_cross(b, v, bv);
            gpst_cross(v, a, va);
            ph[t * edges + e] = w[e] * gpst_dot(v, ab);
            cn[t * edges + e] = w[e] * (gpst_dot(fa, bv) + gpst_dot(fb, va));
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    int64_t q = ((t * edges + e) * 3 + k) * 3 + l;
                    ss[q] = (w[e] * a[k]) * b[l];
                    an[q] = w[e] * (fa[k] * b[l] + a[k] * fb[l]);
                }
            }
        }
    }
    return bad;
}

/* Fused shell-reducing edge kernel: eliminates all O(T * E) intermediate buffers.
 * Each frame t writes strictly to its own disjoint slice -> bit-deterministic across threads. */
int gpst_edge_moments_fused(
    int64_t count, int64_t sites, int64_t edges, int64_t groups,
    const double *s, const double *f,
    const int64_t *left, const int64_t *right, const int64_t *shell,
    const double *w, const double *u,
    double *shell_n,   /* [count, 4, groups] */
    double *shell_d,   /* [count, 4, groups] */
    double *zonal_n,   /* [count] */
    double *zonal_d,   /* [count] */
    double *chir_n,    /* [count] */
    double *chir_d,    /* [count] */
    double *shell_ss,  /* [count, groups, 3, 3] */
    double *shell_an   /* [count, groups, 3, 3] */
) {
    const int orders[4] = {1, 2, 4, 8};
    const double d_factors[4] = {4.0, 12.0, 40.0, 144.0};
    memset(shell_n, 0, (size_t)(count * 4 * groups) * sizeof(double));
    memset(shell_d, 0, (size_t)(count * 4 * groups) * sizeof(double));
    memset(zonal_n, 0, (size_t)count * sizeof(double));
    memset(zonal_d, 0, (size_t)count * sizeof(double));
    memset(chir_n, 0, (size_t)count * sizeof(double));
    memset(chir_d, 0, (size_t)count * sizeof(double));
    memset(shell_ss, 0, (size_t)(count * groups * 9) * sizeof(double));
    memset(shell_an, 0, (size_t)(count * groups * 9) * sizeof(double));

    int bad = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:bad)
#endif
    for (int64_t t = 0; t < count; ++t) {
        double *sn_t = shell_n + t * 4 * groups;
        double *sd_t = shell_d + t * 4 * groups;
        double *ss_t = shell_ss + t * groups * 9;
        double *an_t = shell_an + t * groups * 9;
        double zn_acc = 0.0;
        double zd_acc = 0.0;
        double cn_acc = 0.0;
        double cd_acc = 0.0;

        for (int64_t e = 0; e < edges; ++e) {
            const int64_t g = shell[e];
            if (g < 0 || g >= groups) {
                bad = 1;
                continue;
            }
            const double we = w[e];
            const double *a = s + (t * sites + left[e]) * 3;
            const double *b = s + (t * sites + right[e]) * 3;
            const double *fa = f + (t * sites + left[e]) * 3;
            const double *fb = f + (t * sites + right[e]) * 3;
            const double *v = u + e * 3;

            double x = gpst_dot(a, b);
            if (!isfinite(x) || fabs(x) > 1.0 + 1e-10) {
                bad = 1;
                continue;
            }
            x = fmin(1.0, fmax(-1.0, x));
            double p[9], d[9];
            gpst_poly(x, 8, p, d);

            double c = (b[0] * fa[0] + a[0] * fb[0]) +
                       (b[1] * fa[1] + a[1] * fb[1]) +
                       (b[2] * fa[2] + a[2] * fb[2]);
            for (int k = 0; k < 4; ++k) {
                sn_t[k * groups + g] += (we * d[orders[k]]) * c;
                sd_t[k * groups + g] += d_factors[k] * (we * p[orders[k]]);
            }

            /* Zonal 4th-order probe via exact Horner P4 and P4' for both endpoints. */
            const double *dirs[2] = {a, b};
            const double *forces[2] = {fa, fb};
            for (int k = 0; k < 2; ++k) {
                double z = gpst_dot(dirs[k], v);
                if (!isfinite(z) || fabs(z) > 1.0 + 1e-10) {
                    bad = 1;
                    continue;
                }
                z = fmin(1.0, fmax(-1.0, z));
                double z2 = z * z;
                double p4 = 0.125 * ((35.0 * z2 - 30.0) * z2 + 3.0);
                double dp4 = 0.5 * z * (35.0 * z2 - 15.0);
                double zf = gpst_dot(forces[k], v);
                zn_acc += (we * dp4) * zf;
                zd_acc += 20.0 * (we * p4);
            }

            /* Chirality probe. */
            double ab[3], bv[3], va[3];
            gpst_cross(a, b, ab);
            gpst_cross(b, v, bv);
            gpst_cross(v, a, va);
            cn_acc += we * (gpst_dot(fa, bv) + gpst_dot(fb, va));
            cd_acc += 4.0 * (we * gpst_dot(v, ab));

            /* Shell-resolved 3x3 spin-spin and force-spin tensors. */
            double *ss_g = ss_t + g * 9;
            double *an_g = an_t + g * 9;
            for (int k = 0; k < 3; ++k) {
                double wak = we * a[k];
                double wfak = we * fa[k];
                for (int l = 0; l < 3; ++l) {
                    ss_g[k * 3 + l] += wak * b[l];
                    an_g[k * 3 + l] += wfak * b[l] + wak * fb[l];
                }
            }
        }

        zonal_n[t] = zn_acc;
        zonal_d[t] = zd_acc;
        chir_n[t] = cn_acc;
        chir_d[t] = cd_acc;
    }
    return bad;
}
