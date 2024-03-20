from flask import Flask, render_template, session, url_for, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import numpy as np

app = Flask(__name__)

# Define the Gauss-Jacobi method
def gauss_jacobi(a, b, n=25, tol=1e-10):
    x = np.zeros_like(b)
    for it_count in range(n):
        x_new = np.zeros_like(x)
        for i in range(a.shape[0]):
            s1 = np.dot(a[i, :i], x[:i])
            s2 = np.dot(a[i, i + 1:], x[i + 1:])
            x_new[i] = round((b[i] - s1 - s2) / a[i, i], 4)  # Round to 4 decimal places
        if np.allclose(x, x_new, atol=tol):
            return x_new, it_count
        x = x_new
    return x, n

def is_diagonally_dominant(a):
    # Check if the matrix is diagonally dominant
    D = np.diag(np.abs(a))  # Diagonal elements
    S = np.sum(np.abs(a), axis=1) - D  # Sum of non-diagonal elements
    return np.all(D > S)

def make_diagonally_dominant(a):
    # Attempt to make the matrix diagonally dominant
    # This is a simple heuristic and may not work for all matrices
    for i in range(len(a)):
        row_sum = sum(abs(a[i])) - abs(a[i][i])
        if abs(a[i][i]) < row_sum:
            # Try to find a suitable row to swap with
            for j in range(i+1, len(a)):
                potential_row_sum = sum(abs(a[j])) - abs(a[j][i])
                if abs(a[j][i]) > potential_row_sum:
                    # Swap the rows
                    a[[i, j]] = a[[j, i]]
                    break
    return a

def gauss_seidel(a, b, x0=None, n=25, tol=1e-10):
    # Ensure no diagonal element is zero
    if np.any(np.diag(a) == 0):
        raise ValueError("Diagonal elements cannot be zero.")
    
    # Make the matrix diagonally dominant
    a = make_diagonally_dominant(a)
    
    if x0 is None:
        x0 = np.zeros_like(b)
    x = np.copy(x0)
    
    for it_count in range(n):
        x_new = np.copy(x)
        for i in range(a.shape[0]):
            s1 = np.dot(a[i, :i], x_new[:i])
            s2 = np.dot(a[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / a[i, i]
            x_new[i] = round(x_new[i], 4)  # Round to 4 decimal places
        
        # Check for convergence using relative change
        if np.linalg.norm(x_new - x, ord=np.inf) / np.linalg.norm(x_new, ord=np.inf) < tol:
            return x_new, it_count + 1
        x = x_new
    
    return x, n


bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                session['username'] = form.username.data
                session['sum'] = '25'
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form, name = form.username.data, sum = 25)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    name = session.get('username', 'Guest')
    if request.method == 'POST':
        # Extract coefficients and constants from the form
        a11 = float(request.form['a11'])
        a12 = float(request.form['a12'])
        a13 = float(request.form['a13'])
        a21 = float(request.form['a21'])
        a22 = float(request.form['a22'])
        a23 = float(request.form['a23'])
        a31 = float(request.form['a31'])
        a32 = float(request.form['a32'])
        a33 = float(request.form['a33'])
        b1 = float(request.form['b1'])
        b2 = float(request.form['b2'])
        b3 = float(request.form['b3'])

        # Create numpy arrays for coefficients and constants
        a = np.array([[a11, a12, a13],
                      [a21, a22, a23],
                      [a31, a32, a33]])
        b = np.array([b1, b2, b3])

        # Solve the equations using the Gauss-Jacobi method
        solution, iterations = gauss_jacobi(a, b)
        solution1, iterations1 = gauss_seidel(a, b)
        # Render the solution page
        return render_template('solution.html', solution=solution, iterations=iterations, solution1=solution1, iterations1=iterations1)
    
    # Render the index page with the form
    return render_template('dashboard.html', name=name)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)