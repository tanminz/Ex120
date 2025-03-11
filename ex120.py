import numpy as np


def tao_ma_tran():
    """
    Tạo ma trận ngẫu nhiên NxN, giá trị nguyên trong [-100, 100].
    Trả về ma trận A (ndarray).
    """
    N = int(input("Nhập kích thước ma trận vuông N: "))
    A = np.random.randint(-100, 101, size=(N, N))
    print("Đã tạo ma trận ngẫu nhiên A:")
    print(A)
    return A


def hoan_doi_dong_max_mean(A):
    """
    Tìm dòng có giá trị trung bình lớn nhất rồi hoán đổi với dòng đầu.
    Trả về ma trận A đã được hoán đổi.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return None

    row_means = A.mean(axis=1)  # trung bình mỗi dòng
    max_row_index = np.argmax(row_means)  # chỉ số dòng có trung bình lớn nhất
    if max_row_index != 0:
        A[[0, max_row_index], :] = A[[max_row_index, 0], :]  # hoán đổi dòng
    print("Ma trận sau khi hoán đổi dòng đầu với dòng có mean lớn nhất:")
    print(A)
    return A


def thay_doi_gia_tri_cot(A):
    """
    Yêu cầu người dùng nhập:
      - cột nào muốn thay đổi
      - giá trị c
    Nếu phần tử trong cột < c thì nhân đôi, ngược lại giữ nguyên.
    Trả về ma trận A đã thay đổi.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return None

    col_index = int(input("Nhập chỉ số cột muốn thay đổi (0-based): "))
    c = float(input("Nhập giá trị c: "))

    rows = A.shape[0]
    for i in range(rows):
        if A[i, col_index] < c:
            A[i, col_index] *= 2

    print(f"Ma trận sau khi thay đổi cột {col_index} theo điều kiện với c = {c}:")
    print(A)
    return A


def tinh_dinh_thuc(A):
    """
    Tính và in ra định thức của ma trận A (nếu A là vuông).
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return
    # Vì A là vuông, ta tính được định thức:
    detA = np.linalg.det(A)
    print(f"Định thức của ma trận = {detA}")


def tinh_rank(A):
    """
    Tính và in ra hạng (rank) của ma trận A.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return
    rankA = np.linalg.matrix_rank(A)
    print(f"Hạng (rank) của ma trận = {rankA}")


def tinh_SVD(A):
    """
    Tính và in kết quả phân rã SVD của ma trận A:
      A = U * S * V^T
    Sau đó tái tạo lại ma trận để so sánh.
    """
    if A is None:
        print("Bạn chưa tạo ma trận. Hãy chọn chức năng [1] trước.")
        return

    U, S, Vt = np.linalg.svd(A, full_matrices=True)

    print("Ma trận U:")
    print(U)
    print("Vector singular values (S):")
    print(S)
    print("Ma trận V^T:")
    print(Vt)

    # Tái tạo ma trận A từ U, S, V^T
    N = A.shape[0]  # vì A là NxN
    S_new = np.zeros((N, N))
    for i in range(len(S)):
        S_new[i, i] = S[i]
    A_reconstructed = U @ S_new @ Vt

    print("Ma trận tái tạo (U * S * V^T):")
    print(A_reconstructed)

    # Tính sai khác
    diff = np.abs(A - A_reconstructed)
    print("Sai khác giữa A và A_reconstructed (giá trị tuyệt đối):")
    print(diff)
    print("Tổng sai khác =", np.sum(diff))


def main():
    A = None  # ban đầu chưa có ma trận
    while True:
        print("\n===== MENU =====")
        print("1. Tạo ma trận ngẫu nhiên NxN")
        print("2. Tìm dòng có giá trị trung bình lớn nhất và hoán đổi với dòng đầu")
        print("3. Thay đổi giá trị của một cột theo điều kiện")
        print("4. Tính định thức của ma trận")
        print("5. Tính hạng (rank) của ma trận")
        print("6. Tính SVD của ma trận")
        print("7. Thoát")
        print("================")

        choice = input("Chọn chức năng (1-7): ").strip()

        if choice == "1":
            A = tao_ma_tran()
        elif choice == "2":
            A = hoan_doi_dong_max_mean(A)
        elif choice == "3":
            A = thay_doi_gia_tri_cot(A)
        elif choice == "4":
            tinh_dinh_thuc(A)
        elif choice == "5":
            tinh_rank(A)
        elif choice == "6":
            tinh_SVD(A)
        elif choice == "7":
            print("Kết thúc chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")


if __name__ == "__main__":
    main()
