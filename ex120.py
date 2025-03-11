import numpy as np


def create_matrix():
    """
    Tạo ma trận ngẫu nhiên có kích thước m x n với giá trị nguyên trong khoảng [-100, 100].
    """
    m = int(input("Nhập số dòng m: "))
    n = int(input("Nhập số cột n: "))
    A = np.random.randint(-100, 101, size=(m, n))
    print("Ma trận A:")
    print(A)
    return A


def swap_row_with_max_mean(A):
    """
    Tìm dòng có giá trị trung bình lớn nhất và hoán đổi với dòng đầu.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return None

    row_means = A.mean(axis=1)  # Tính trung bình của từng dòng
    max_index = np.argmax(row_means)

    if max_index != 0:
        A[[0, max_index], :] = A[[max_index, 0], :]

    print("Ma trận sau khi hoán đổi dòng đầu với dòng có mean lớn nhất:")
    print(A)
    return A


def change_column_values(A):
    """
    Thay đổi giá trị của một cột theo điều kiện:
    Nếu phần tử trong cột đó nhỏ hơn giá trị c thì nhân đôi.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return None

    col_index = int(input("Nhập chỉ số cột muốn thay đổi (0-based): "))
    c = float(input("Nhập giá trị c: "))

    for i in range(A.shape[0]):
        if A[i, col_index] < c:
            A[i, col_index] *= 2

    print(f"Ma trận sau khi thay đổi cột {col_index} theo điều kiện với c = {c}:")
    print(A)
    return A


def compute_determinant(A):
    """
    Tính định thức của ma trận nếu nó là ma trận vuông.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return
    m, n = A.shape
    if m != n:
        print("Không thể tính định thức cho ma trận không vuông (m ≠ n).")
        return
    det = np.linalg.det(A)
    print("Định thức của ma trận A =", det)


def compute_rank(A):
    """
    Tính và in ra hạng (rank) của ma trận.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return
    r = np.linalg.matrix_rank(A)
    print("Hạng (rank) của ma trận A =", r)


def compute_SVD(A):
    """
    Tính phân rã SVD của ma trận A: A = U * S * V^T, sau đó tái tạo lại ma trận.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return

    # Sử dụng economy SVD để dễ dàng tái tạo lại A
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    print("Ma trận U:")
    print(U)
    print("Vector singular values (S):")
    print(S)
    print("Ma trận V^T:")
    print(Vt)

    # Tái tạo ma trận A từ U, S và V^T
    A_reconstructed = U @ np.diag(S) @ Vt
    print("Ma trận tái tạo từ SVD:")
    print(A_reconstructed)

    # Tính sai khác giữa ma trận ban đầu và ma trận tái tạo
    diff = np.abs(A - A_reconstructed)
    print("Sai khác giữa A và A_reconstructed:")
    print(diff)
    print("Tổng sai khác =", np.sum(diff))


def main():
    A = None  # Khởi tạo biến ma trận ban đầu
    while True:
        print("\n===== MENU =====")
        print("1. Tạo ma trận ngẫu nhiên m x n")
        print("2. Hoán đổi dòng có giá trị trung bình lớn nhất với dòng đầu")
        print("3. Thay đổi giá trị của một cột theo điều kiện")
        print("4. Tính định thức (nếu là ma trận vuông)")
        print("5. Tính hạng (rank) của ma trận")
        print("6. Tính SVD của ma trận và tái tạo lại ma trận")
        print("7. Thoát")
        print("================")

        choice = input("Chọn chức năng (1-7): ").strip()

        if choice == "1":
            A = create_matrix()
        elif choice == "2":
            A = swap_row_with_max_mean(A)
        elif choice == "3":
            A = change_column_values(A)
        elif choice == "4":
            compute_determinant(A)
        elif choice == "5":
            compute_rank(A)
        elif choice == "6":
            compute_SVD(A)
        elif choice == "7":
            print("Kết thúc chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")


if __name__ == "__main__":
    main()
