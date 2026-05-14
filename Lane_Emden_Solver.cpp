/////////////////////////////////////////////////////////////////////////
/////////////// Lane-Emden Equation Solver - Polytropes /////////////////
/////////////////////////////////////////////////////////////////////////
//
// Lorenzo Monti
//
// 05/2026
//

#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <fstream>

using namespace std;

#define NMAX        200000
#define NMODELS     5

// Physical constants
const double G       = 6.67430e-11;          // Gravitational constant (m^3 kg^-1 s^-2)
const double M_sun   = 1.98847e30;           // Solar mass (kg)
const double M_p     = 1.67262192369e-27;    // Proton mass (kg)
const double HBAR    = 1.054571817e-34;      // Reduced Planck constant (J s)
const double C_light = 2.99792458e8;         // Speed of light in vacuum (m s^-1)

// Global arrays
double solution_xi_values[NMODELS][NMAX];
double solution_theta_values[NMODELS][NMAX];
double solution_phi_values[NMODELS][NMAX];

double plot_xi_values[NMODELS][NMAX];
double plot_theta_values[NMODELS][NMAX];
double plot_phi_values[NMODELS][NMAX];

// Function prototypes
void LaneEmdenRHS(double xi, double theta, double phi, double n, double &dtheta, double &dphi);
void RK4Step(double xi, double theta, double phi, double h, double n,
             double &xi_next, double &theta_next, double &phi_next);

double FindZeroLinear(double xi_old, double theta_old, double xi_new, double theta_new);

void SolveLaneEmden(double n, double xi_max, double h, bool stop_at_zero,
                    double xi_values[], double theta_values[], double phi_values[],
                    int &n_points, double &xi_zero, bool &has_zero);

double ThetaExact(double xi, double n);
bool ExactFirstZero(double n, double &xi_exact);
double ValidationError(double n, double xi[], double theta[], int n_points, bool has_zero);

bool DimensionlessMass(double phi[], int n_points, double xi_zero, bool has_zero, double &mu);

double WhiteDwarfK();
void ChandrasekharMass(double n, double phi[], int n_points, double xi_zero, bool has_zero,
                       double &K, double &M_ch, double &M_ch_solar);

void PrintChandrasekharSummary(double n, double phi[], int n_points,
                               double xi_zero, bool has_zero);

void PrintSummary(double n_values[],
                  double xi_values[][NMAX],
                  double theta_values[][NMAX],
                  double phi_values[][NMAX],
                  int n_points[],
                  double xi_zero_values[],
                  bool has_zero_values[],
                  int n_models);

void WriteSolutionData(double xi_values[][NMAX],
                       double theta_values[][NMAX],
                       int n_points[],
                       int n_models,
                       const char filename[]);

void PlotSolution(double n_values[],
                  double xi_values[][NMAX],
                  double theta_values[][NMAX],
                  int n_points[],
                  int n_models);

void PlotDimensionlessDensityProfiles(double n_values[],
                                      double xi_values[][NMAX],
                                      double theta_values[][NMAX],
                                      int n_points[],
                                      double xi_zero_values[],
                                      bool has_zero_values[],
                                      int n_models);


// MAIN FUNCTION
int main()
{
    double n_values[NMODELS] = {0.0, 1.0, 1.5, 3.0, 5.0};

    double xi_max = 10.0;
    double h = 1.0e-4;

    int solution_n_points[NMODELS];
    double solution_xi_zero_values[NMODELS];
    bool solution_has_zero_values[NMODELS];

    int plot_n_points[NMODELS];
    double plot_xi_zero_values[NMODELS];
    bool plot_has_zero_values[NMODELS];

    cout << "=== Lane-Emden solver ===" << endl << endl;

    for (int i = 0; i < NMODELS; i++) {
        double n = n_values[i];

        SolveLaneEmden(n, xi_max, h, true,
                       solution_xi_values[i],
                       solution_theta_values[i],
                       solution_phi_values[i],
                       solution_n_points[i],
                       solution_xi_zero_values[i],
                       solution_has_zero_values[i]);

        SolveLaneEmden(n, xi_max, h, false,
                       plot_xi_values[i],
                       plot_theta_values[i],
                       plot_phi_values[i],
                       plot_n_points[i],
                       plot_xi_zero_values[i],
                       plot_has_zero_values[i]);
    }

    PrintSummary(n_values,
                 solution_xi_values,
                 solution_theta_values,
                 solution_phi_values,
                 solution_n_points,
                 solution_xi_zero_values,
                 solution_has_zero_values,
                 NMODELS);

    PlotSolution(n_values,
                 plot_xi_values,
                 plot_theta_values,
                 plot_n_points,
                 NMODELS);

    PlotDimensionlessDensityProfiles(n_values,
                                     solution_xi_values,
                                     solution_theta_values,
                                     solution_n_points,
                                     solution_xi_zero_values,
                                     solution_has_zero_values,
                                     NMODELS);

    for (int i = 0; i < NMODELS; i++) {
        if (n_values[i] == 3.0) {
            PrintChandrasekharSummary(n_values[i],
                                      solution_phi_values[i],
                                      solution_n_points[i],
                                      solution_xi_zero_values[i],
                                      solution_has_zero_values[i]);
            break;
        }
    }

    return 0;
}


// Function to compute the right-hand side of the Lane-Emden system
void LaneEmdenRHS(double xi, double theta, double phi, double n, double &dtheta, double &dphi)
{
    double theta_power;

    // Fractional powers of negative theta are not real-valued.
    if (theta < 0.0 && abs(n - round(n)) > 1.0e-12) {
        theta_power = 0.0;
    } else {
        theta_power = theta;
    }

    dtheta = phi;

    if (xi == 0.0) {
        dphi = 0.0;
    } else {
        dphi = -2.0 * phi / xi - pow(theta_power, n);
    }
}


// Runge-Kutta 4th-order step for the Lane-Emden system
void RK4Step(double xi, double theta, double phi, double h, double n,
             double &xi_next, double &theta_next, double &phi_next)
{
    double k1_theta, k1_phi;
    double k2_theta, k2_phi;
    double k3_theta, k3_phi;
    double k4_theta, k4_phi;

    LaneEmdenRHS(xi, theta, phi, n, k1_theta, k1_phi);

    LaneEmdenRHS(xi + 0.5 * h,
                 theta + 0.5 * h * k1_theta,
                 phi   + 0.5 * h * k1_phi,
                 n,
                 k2_theta,
                 k2_phi);

    LaneEmdenRHS(xi + 0.5 * h,
                 theta + 0.5 * h * k2_theta,
                 phi   + 0.5 * h * k2_phi,
                 n,
                 k3_theta,
                 k3_phi);

    LaneEmdenRHS(xi + h,
                 theta + h * k3_theta,
                 phi   + h * k3_phi,
                 n,
                 k4_theta,
                 k4_phi);

    xi_next = xi + h;

    theta_next = theta + h / 6.0 * (
        k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta
    );

    phi_next = phi + h / 6.0 * (
        k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi
    );
}


// Function to estimate the first zero of theta
double FindZeroLinear(double xi_old, double theta_old, double xi_new, double theta_new)
{
    return xi_old - theta_old * (xi_new - xi_old) / (theta_new - theta_old);
}


// Function to solve the Lane-Emden equation
void SolveLaneEmden(double n, double xi_max, double h, bool stop_at_zero,
                    double xi_values[], double theta_values[], double phi_values[],
                    int &n_points, double &xi_zero, bool &has_zero)
{
    double xi = h;

    // Taylor expansion near the origin
    double theta = 1.0 - xi * xi / 6.0 + n * pow(xi, 4) / 120.0;
    double phi   = -xi / 3.0 + n * pow(xi, 3) / 30.0;

    n_points = 0;
    xi_zero = 0.0;
    has_zero = false;

    xi_values[n_points] = xi;
    theta_values[n_points] = theta;
    phi_values[n_points] = phi;
    n_points++;

    while (xi < xi_max && n_points < NMAX) {
        double xi_old = xi;
        double theta_old = theta;
        double phi_old = phi;

        double xi_next;
        double theta_next;
        double phi_next;

        RK4Step(xi, theta, phi, h, n, xi_next, theta_next, phi_next);

        xi = xi_next;
        theta = theta_next;
        phi = phi_next;

        // The first zero of theta defines the physical stellar surface.
        if (theta <= 0.0 && !has_zero) {
            xi_zero = FindZeroLinear(xi_old, theta_old, xi, theta);
            has_zero = true;

            double phi_zero = phi_old + (phi - phi_old) *
                              (xi_zero - xi_old) / (xi - xi_old);

            if (stop_at_zero) {
                xi_values[n_points] = xi_zero;
                theta_values[n_points] = 0.0;
                phi_values[n_points] = phi_zero;
                n_points++;
                break;
            }
        }

        xi_values[n_points] = xi;
        theta_values[n_points] = theta;
        phi_values[n_points] = phi;
        n_points++;
    }
}


// Function to return analytical Lane-Emden solutions
double ThetaExact(double xi, double n)
{
    if (n == 0.0) {
        return 1.0 - xi * xi / 6.0;
    }

    if (n == 1.0) {
        return sin(xi) / xi;
    }

    if (n == 5.0) {
        return 1.0 / sqrt(1.0 + xi * xi / 3.0);
    }

    return NAN;
}


// Function to return the exact first zero when available
bool ExactFirstZero(double n, double &xi_exact)
{
    if (n == 0.0) {
        xi_exact = sqrt(6.0);
        return true;
    }

    if (n == 1.0) {
        xi_exact = M_PI;
        return true;
    }

    return false;
}


// Function to compute the maximum absolute validation error
double ValidationError(double n, double xi[], double theta[], int n_points, bool has_zero)
{
    int size = n_points;

    // The final interpolated zero is not an RK4 integration point.
    if (has_zero && size > 0) {
        size--;
    }

    double max_error = 0.0;

    for (int i = 0; i < size; i++) {
        double theta_reference = ThetaExact(xi[i], n);

        if (isnan(theta_reference)) {
            return NAN;
        }

        double error = abs(theta[i] - theta_reference);

        if (error > max_error) {
            max_error = error;
        }
    }

    return max_error;
}


// Function to compute the dimensionless mass
bool DimensionlessMass(double phi[], int n_points, double xi_zero, bool has_zero, double &mu)
{
    if (!has_zero || n_points <= 0) {
        return false;
    }

    double phi_1 = phi[n_points - 1];
    mu = -xi_zero * xi_zero * phi_1;

    return true;
}


// Function to compute the relativistic white dwarf polytropic constant
double WhiteDwarfK()
{
    return (
        (pow(3.0, 1.0 / 3.0) * pow(M_PI, 2.0 / 3.0))
        / (pow(2.0, 4.0 / 3.0) * 4.0)
        * HBAR
        * C_light
        / pow(M_p, 4.0 / 3.0)
    );
}


// Function to compute the Chandrasekhar mass
void ChandrasekharMass(double n, double phi[], int n_points, double xi_zero, bool has_zero,
                       double &K, double &M_ch, double &M_ch_solar)
{
    if (n != 3.0) {
        cout << "Error: Chandrasekhar mass calculation requires n = 3." << endl;
        exit(1);
    }

    double mu;

    if (!DimensionlessMass(phi, n_points, xi_zero, has_zero, mu)) {
        cout << "Error: missing finite zero for Chandrasekhar mass calculation." << endl;
        exit(1);
    }

    K = WhiteDwarfK();

    M_ch = 4.0 * M_PI * mu *
           pow(((n + 1.0) * K) / (4.0 * M_PI * G), 1.5);

    M_ch_solar = M_ch / M_sun;
}


// Function to print the Chandrasekhar mass summary
void PrintChandrasekharSummary(double n, double phi[], int n_points,
                               double xi_zero, bool has_zero)
{
    double K;
    double M_ch;
    double M_ch_solar;

    ChandrasekharMass(n, phi, n_points, xi_zero, has_zero, K, M_ch, M_ch_solar);

    cout << endl;
    cout << "High-density white dwarf model" << endl;
    cout << "--------------------------------" << endl;

    cout << scientific << setprecision(6);
    cout << "K = " << K << " SI" << endl;
    cout << "M_Ch = " << M_ch << " kg" << endl;

    cout << fixed << setprecision(6);
    cout << "M_Ch = " << M_ch_solar << " M_sun" << endl;
}


// Function to print the dimensionless summary table
void PrintSummary(double n_values[],
                  double xi_values[][NMAX],
                  double theta_values[][NMAX],
                  double phi_values[][NMAX],
                  int n_points[],
                  double xi_zero_values[],
                  bool has_zero_values[],
                  int n_models)
{
    cout << "Dimensionless summary" << endl;
    cout << "n        xi_1             mu              max_error        zero_error" << endl;
    cout << "---------------------------------------------------------------------" << endl;

    for (int i = 0; i < n_models; i++) {
        double n = n_values[i];
        double mu;
        double max_error;
        double xi_exact;
        double zero_error;

        cout << fixed << setprecision(1)
             << left << setw(9) << n;

        if (!DimensionlessMass(phi_values[i], n_points[i],
                               xi_zero_values[i],
                               has_zero_values[i],
                               mu)) {
            cout << setw(17) << "None"
                 << setw(16) << "None";
        } else {
            cout << fixed << setprecision(10)
                 << setw(17) << xi_zero_values[i]
                 << setw(16) << mu;
        }

        max_error = ValidationError(n, xi_values[i], theta_values[i],
                                    n_points[i], has_zero_values[i]);

        if (isnan(max_error)) {
            cout << setw(16) << "None";
        } else {
            cout << scientific << setprecision(3)
                 << setw(16) << max_error;
        }

        if (has_zero_values[i] && ExactFirstZero(n, xi_exact)) {
            zero_error = abs(xi_zero_values[i] - xi_exact);
            cout << scientific << setprecision(3) << zero_error;
        } else {
            cout << "None";
        }

        cout << endl;
    }
}


// Function to write Lane-Emden solution data
void WriteSolutionData(double xi_values[][NMAX],
                       double theta_values[][NMAX],
                       int n_points[],
                       int n_models,
                       const char filename[])
{
    ofstream file(filename);

    for (int i = 0; i < n_models; i++) {
        for (int j = 0; j < n_points[i]; j++) {
            file << xi_values[i][j] << " "
                 << theta_values[i][j] << endl;
        }

        file << endl << endl;
    }

    file.close();
}


// Function to plot the Lane-Emden solutions
void PlotSolution(double n_values[],
                  double xi_values[][NMAX],
                  double theta_values[][NMAX],
                  int n_points[],
                  int n_models)
{
    system("mkdir -p Images");

    const char data_filename[] = "Images/lane_emden.dat";
    const char script_filename[] = "Images/lane_emden.gnuplot";
    const char output_filename[] = "Images/lane_emden.pdf";

    WriteSolutionData(xi_values, theta_values, n_points, n_models, data_filename);

    ofstream script(script_filename);

    script << "set terminal pdfcairo enhanced color" << endl;
    script << "set output '" << output_filename << "'" << endl;
    script << "set xlabel '{/Symbol x}'" << endl;
    script << "set ylabel '{/Symbol q}'" << endl;
    script << "set xrange [0:10]" << endl;
    script << "set yrange [-0.5:1.0]" << endl;
    script << "set grid" << endl;
    script << "set key" << endl;
    script << "set arrow from 0,0 to 10,0 nohead lc rgb 'black' lw 0.8" << endl;

    script << "set title 'Lane-Emden solutions (n = ";

    for (int i = 0; i < n_models; i++) {
        script << n_values[i];

        if (i + 1 < n_models) {
            script << ", ";
        }
    }

    script << ")'" << endl;

    script << "plot ";

    for (int i = 0; i < n_models; i++) {
        script << "'" << data_filename << "' index " << i
               << " using 1:2 with lines title 'n = " << n_values[i] << "'";

        if (i + 1 < n_models) {
            script << ", ";
        }
    }

    script << endl;
    script.close();

    system("gnuplot Images/lane_emden.gnuplot");

    cout << endl;
    cout << "Plot saved as: " << output_filename << endl;
}


// Function to plot dimensionless density profiles
void PlotDimensionlessDensityProfiles(double n_values[],
                                      double xi_values[][NMAX],
                                      double theta_values[][NMAX],
                                      int n_points[],
                                      double xi_zero_values[],
                                      bool has_zero_values[],
                                      int n_models)
{
    system("mkdir -p Images");

    const char data_filename[] = "Images/dimensionless_density_profiles.dat";
    const char script_filename[] = "Images/dimensionless_density_profiles.gnuplot";
    const char output_filename[] = "Images/dimensionless_density_profiles.pdf";

    ofstream data(data_filename);

    double plotted_n_values[NMODELS];
    int n_plotted = 0;

    for (int i = 0; i < n_models; i++) {
        double n = n_values[i];

        if (!(n == 0.0 || n == 1.0 || n == 1.5 || n == 3.0)) {
            continue;
        }

        if (!has_zero_values[i]) {
            continue;
        }

        double xi_1 = xi_zero_values[i];

        for (int j = 0; j < n_points[i]; j++) {
            double theta = theta_values[i][j];

            if (theta < 0.0) {
                theta = 0.0;
            }

            double density;

            if (n == 0.0) {
                density = 1.0;

                if (j + 1 == n_points[i]) {
                    density = 0.0;
                }
            } else {
                density = pow(theta, n);
            }

            data << xi_values[i][j] / xi_1 << " "
                 << density << endl;
        }

        data << endl << endl;

        plotted_n_values[n_plotted] = n;
        n_plotted++;
    }

    data.close();

    ofstream script(script_filename);

    script << "set terminal pdfcairo enhanced color" << endl;
    script << "set output '" << output_filename << "'" << endl;
    script << "set xlabel 'r/R = {/Symbol x}/{/Symbol x}_1'" << endl;
    script << "set ylabel '{/Symbol r}/{/Symbol r}_c'" << endl;
    script << "set title 'Dimensionless density profiles'" << endl;
    script << "set grid" << endl;
    script << "set key" << endl;

    script << "plot ";

    for (int i = 0; i < n_plotted; i++) {
        script << "'" << data_filename << "' index " << i
               << " using 1:2 with lines title 'n = " << plotted_n_values[i] << "'";

        if (i + 1 < n_plotted) {
            script << ", ";
        }
    }

    script << endl;
    script.close();

    system("gnuplot Images/dimensionless_density_profiles.gnuplot");

    cout << "Density plot saved as: " << output_filename << endl;
}
