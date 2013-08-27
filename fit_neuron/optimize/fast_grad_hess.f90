subroutine fast_grad_hess(thresh_param,X_cat,param_ct,ind_ct,grad_vec,hess_mat)
    implicit none
    integer, intent(in):: ind_ct,param_ct
    double precision, dimension(param_ct,1), intent(in) :: thresh_param
    double precision, dimension(ind_ct,param_ct), intent(in) :: X_cat
    double precision, dimension(param_ct,param_ct), intent(out) :: hess_mat
    double precision, dimension(param_ct,1), intent(out) :: grad_vec

    ! Declarations for temp variables
    double precision, dimension(param_ct,1) :: X_thresh
    double precision :: exp_val
    integer t


    do t=1,ind_ct
        X_thresh(:,1) = X_cat(t,:)

        if (isnan(X_thresh(1,1))) then
            cycle
        end if

        if (t .le. ind_ct) then
            if (isnan(X_cat(t+1,1))) then
                grad_vec = grad_vec + X_thresh
                cycle
            end if
        end if

        exp_val = exp(dot_product(thresh_param(:,1),X_thresh(:,1)))
        grad_vec = grad_vec - X_thresh * exp_val
        hess_mat = hess_mat - exp_val * matmul(X_thresh,transpose(X_thresh))
    end do
end subroutine
