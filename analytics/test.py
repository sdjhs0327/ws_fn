import random
import pygame

# 게임 화면 크기 설정
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# 벽돌 크기 설정
BRICK_WIDTH = 60
BRICK_HEIGHT = 20

# 공 크기 설정
BALL_RADIUS = 10

# 색상 설정
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# 벽돌 클래스 정의
class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([BRICK_WIDTH, BRICK_HEIGHT])
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

# 공 클래스 정의
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([BALL_RADIUS * 2, BALL_RADIUS * 2])
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.centerx = SCREEN_WIDTH // 2
        self.rect.centery = SCREEN_HEIGHT // 2
        self.speed_x = random.choice([-2, 2])
        self.speed_y = -2

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
            self.speed_x *= -1
        if self.rect.top < 0:
            self.speed_y *= -1

# 패들 클래스 정의
class Paddle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface([100, 20])
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH // 2 - 50
        self.rect.y = SCREEN_HEIGHT - 40
        self.speed = 5

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

# 게임 초기화
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("벽돌깨기 게임")

# 벽돌 그룹 생성
bricks = pygame.sprite.Group()

# 벽돌 생성
for row in range(5):
    for column in range(10):
        brick = Brick(column * (BRICK_WIDTH + 5) + 30, row * (BRICK_HEIGHT + 5) + 30)
        bricks.add(brick)

# 패들 생성
paddle = Paddle()

# 공 생성
ball = Ball()

# 스프라이트 그룹에 패들과 공 추가
all_sprites = pygame.sprite.Group()
all_sprites.add(paddle, ball)

# 게임 루프
running = True
score = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 화면 초기화
    screen.fill(WHITE)

    # 패들 업데이트
    all_sprites.update()

    # 벽돌과 공의 충돌 체크
    brick_collision = pygame.sprite.spritecollide(ball, bricks, True)
    if brick_collision:
        score += 1

    # 벽돌 그리기
    bricks.draw(screen)

    # 패들과 공 그리기
    all_sprites.draw(screen)

    # 점수 표시
    font = pygame.font.Font(None, 36)
    score_text = font.render("Score: " + str(score), True, BLUE)
    screen.blit(score_text, (10, 10))

    # 화면 업데이트
    pygame.display.flip()

# 게임 종료
pygame.quit()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# 벽돌 크기 설정
BRICK_WIDTH = 60
BRICK_HEIGHT = 20

# 공 크기 설정
BALL_RADIUS = 10

# 색상 설정
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# 벽돌 클래스 정의
class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BRICK_WIDTH, BRICK_HEIGHT))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

# 공 클래스 정의
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BALL_RADIUS * 2, BALL_RADIUS * 2))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.velocity = [3, 3]  # Initial velocity of the ball

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

        # Check collision with walls
        if self.rect.left <= 0 or self.rect.right >= SCREEN_WIDTH:
            self.velocity[0] = -self.velocity[0]
        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.velocity[1] = -self.velocity[1]

        # Check collision with paddle
        if self.rect.colliderect(paddle.rect):
            self.velocity[1] = -self.velocity[1]

# 패들 클래스 정의
class Paddle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((100, 20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50)
        self.velocity = [0, 0]  # Initial velocity of the paddle

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

        # Limit paddle movement within the screen
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

# 게임 초기화
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("벽돌깨기 게임")

# 벽돌 그룹 생성
bricks = pygame.sprite.Group()

# 벽돌 생성
for row in range(5):
    for column in range(10):
        brick = Brick(column * (BRICK_WIDTH + 5) + 30, row * (BRICK_HEIGHT + 5) + 30)
        bricks.add(brick)

# 패들 생성
paddle = Paddle()

# 공 생성
ball = Ball()

# 스프라이트 그룹에 패들과 공 추가
all_sprites = pygame.sprite.Group()
all_sprites.add(paddle, ball)

# 게임 루프
running = True
score = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                paddle.velocity[0] = -5
            elif event.key == pygame.K_RIGHT:
                paddle.velocity[0] = 5
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                paddle.velocity[0] = 0

    # 화면 초기화
    screen.fill(WHITE)

    # 패들 업데이트
    all_sprites.update()

    # 벽돌과 공의 충돌 체크
    brick_collision = pygame.sprite.spritecollide(ball, bricks, True)
    if brick_collision:
        score += 1

    # 벽돌 그리기
    bricks.draw(screen)

    # 패들과 공 그리기
    all_sprites.draw(screen)

    # 점수 표시
    font = pygame.font.Font(None, 36)
    score_text = font.render("Score: " + str(score), True, BLUE)
    screen.blit(score_text, (10, 10))

    # 화면 업데이트
    pygame.display.flip()

# 게임 종료
pygame.quit()

