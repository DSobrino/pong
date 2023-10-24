<template>
  <div class="pong__app">
    <Pong class="pong__game" width="600" height="400" />
  </div>
</template>

<script setup lang="ts">
import Pong from './components/Pong.vue';
</script>

<style scoped lang="scss">
.pong {
  &__app {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  &__game {
    border: 1px solid white;
  }
}
</style>
