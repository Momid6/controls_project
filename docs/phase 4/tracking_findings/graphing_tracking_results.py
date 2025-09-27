import numpy as np
import matplotlib.pyplot as plt

x_axis = np.arange(1, 502, 1)
convex_mpc_data = np.loadtxt('tracking_error_log_convex_mpc.txt')
tiny_mpc_data = np.loadtxt('tracking_error_log_tiny_mpc.txt')
fhlqr_data = np.loadtxt('tracking_error_log_fhlqr.txt')
fhlqr_data = np.append(fhlqr_data, fhlqr_data[-1])
ihlqr_data = np.loadtxt('tracking_error_log_ihlqr.txt')
pid_data = np.loadtxt('tracking_error_log_pid.txt')

plt.plot(x_axis, convex_mpc_data, label="convex_mpc")
plt.plot(x_axis, tiny_mpc_data, label="tiny_mpc")
plt.plot(x_axis, fhlqr_data, label="fhlqr")
plt.plot(x_axis, ihlqr_data, label="ihlqr")
plt.plot(x_axis, pid_data, label="pid")
plt.xlabel('Time')
plt.legend()
plt.ylabel('Average Tracking Error L2 Norm')
plt.title('Tracking Error Over Time')
plt.grid(True)
plt.show()

