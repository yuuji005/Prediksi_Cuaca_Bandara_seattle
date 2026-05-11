document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.querySelector('form');
    const submitBtn = document.querySelector('button[type="submit"]');

    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            // Cek apakah form valid sebelum menampilkan loading
            if (predictionForm.checkValidity()) {
                // Ubah teks dan disable tombol untuk mencegah multiple clicks
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Model Sedang Memproses...';
                submitBtn.disabled = true;
                submitBtn.classList.add('opacity-75');
            }
        });
    }
});