all: src/main.rs
	cargo run

src/main.rs: main.rs.tpl
	cp main.rs.tpl src/main.rs

cleandir:
	rm -f src/main.rs
